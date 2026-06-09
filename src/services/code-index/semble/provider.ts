import * as path from "path"
import * as vscode from "vscode"

import { IndexingState } from "../interfaces/manager"
import { VectorStoreSearchResult } from "../interfaces/vector-store"
import { CodeIndexStateManager } from "../state-manager"
import { SembleCLI } from "./semble-cli"
import { downloadSemble, isSembleSupportedPlatform } from "./semble-downloader"
import { ISembleProvider, SembleConfig, SembleContentType, SembleSearchResult, SEMBLE_DEFAULTS } from "./types"
import { TelemetryService } from "@roo-code/telemetry"
import { TelemetryEventName } from "@roo-code/types"
import { t } from "../../../i18n"

/**
 * Result from _ensureModelReady — either success or failure with error details.
 */
type ModelReadinessResult =
	| { success: true; wasRepaired?: boolean }
	| { success: false; error: string; wasRepaired: boolean }

/**
 * Orchestrates code search via the semble CLI.
 *
 * Semble indexes on-the-fly with each search call — there is no separate
 * "indexing" step. The provider automatically downloads the semble binary
 * on first use, then delegates search queries to `semble search`.
 *
 * When `embedderProvider === "semble"`, the CodeIndexManager delegates
 * to this provider instead of the ServiceFactory → orchestrator pipeline.
 */
export class SembleProvider implements ISembleProvider {
	private cli!: SembleCLI
	private readonly workspacePath: string
	private readonly config: SembleConfig
	private readonly stateManager: CodeIndexStateManager
	private readonly context: vscode.ExtensionContext

	private _state: IndexingState = "Standby"
	private _isInitialized = false
	private _initPromise: Promise<void> | undefined

	constructor(
		workspacePath: string,
		context: vscode.ExtensionContext,
		stateManager: CodeIndexStateManager,
		options?: { topK?: number; content?: SembleContentType },
	) {
		this.workspacePath = workspacePath
		this.context = context
		this.stateManager = stateManager

		this.config = {
			topK: options?.topK ?? SEMBLE_DEFAULTS.DEFAULT_TOP_K,
			content: options?.content ?? SEMBLE_DEFAULTS.DEFAULT_CONTENT,
		}
	}

	get state(): IndexingState {
		return this._state
	}

	async initialize(): Promise<void> {
		if (this._isInitialized) {
			return
		}

		if (this._initPromise) {
			return this._initPromise
		}

		this._initPromise = this._doInitialize()
		try {
			await this._initPromise
		} finally {
			this._initPromise = undefined
		}
	}

	private async _doInitialize(): Promise<void> {
		if (!isSembleSupportedPlatform()) {
			this._state = "Error"
			this.stateManager.setSystemState(
				"Error",
				`Semble is not supported on this platform (${process.platform}-${process.arch}).`,
			)
			console.error(`[SembleProvider] Unsupported platform: ${process.platform}-${process.arch}`)
			return
		}

		try {
			this.stateManager.setSystemState("Indexing", t("embeddings:semble.downloadingBinary"))
			const storageDir = this.context.globalStorageUri.fsPath
			const binaryPath = await downloadSemble(storageDir)
			if (!binaryPath) {
				throw new Error("Download returned no path")
			}
			this.cli = new SembleCLI(binaryPath)
		} catch (error: any) {
			this._state = "Error"
			this.stateManager.setSystemState("Error", `Failed to download semble: ${error?.message || error}`)
			console.error("[SembleProvider] Download failed:", error?.message || error)
			return
		}

		const checkResult = await this.cli.checkInstalled()

		if (!checkResult.installed) {
			const errorMsg = checkResult.error || "Semble binary is not functional"
			this._state = "Error"
			this.stateManager.setSystemState("Error", `Semble check failed: ${errorMsg}`)
			console.error("[SembleProvider] Semble check failed:", errorMsg)
			return
		}

		console.log("[SembleProvider] Semble binary found and ready. Verifying embedding model...")

		const modelResult = await this._ensureModelReady()

		if (!modelResult.success) {
			this._state = "Error"
			this.stateManager.setSystemState(
				"Error",
				t("embeddings:semble.downloadFailed", { error: modelResult.error }),
			)
			console.error("[SembleProvider] Model check failed:", modelResult.error)
			return
		}

		if (modelResult.wasRepaired) {
			console.log("[SembleProvider] Model was corrupted and has been re-downloaded successfully.")
		}

		console.log("[SembleProvider] Embedding model verified and ready.")

		this._state = "Indexed"
		this.stateManager.setSystemState("Indexed", t("embeddings:semble.ready"))

		this._isInitialized = true
	}

	private async _ensureModelReady(): Promise<ModelReadinessResult> {
		// Step 1: Run smoke search via CLI to verify model is functional.
		// This also triggers first-time model download if needed.
		this.stateManager.setSystemState("Indexing", t("embeddings:semble.verifyingModel"))
		const smokeResult = await this.cli.checkModel()

		if (smokeResult.installed) {
			return { success: true }
		}

		const errorMsg = smokeResult.error || ""
		if (errorMsg.toLowerCase().includes("timeout") || errorMsg.toLowerCase().includes("timed out")) {
			console.warn(`[SembleProvider] Model smoke search timed out (transient): ${errorMsg}`)
			return {
				success: false,
				error: `Model loading timed out. This is normal on first use. Please try again.`,
				wasRepaired: false,
			}
		}

		console.warn(`[SembleProvider] Model verification failed: ${errorMsg}`)

		// Step 2: Attempt repair — clear cache and re-download
		console.log("[SembleProvider] Clearing corrupted model cache...")
		this.stateManager.setSystemState("Indexing", t("embeddings:semble.clearingCorruptedCache"))

		const clearResult = this.cli.clearModelCache()
		if (!clearResult.cleared) {
			return {
				success: false,
				error: `Failed to clear corrupted model cache: ${clearResult.error}`,
				wasRepaired: false,
			}
		}

		console.log("[SembleProvider] Downloading embedding model from HuggingFace...")
		this.stateManager.setSystemState("Indexing", t("embeddings:semble.downloadingModel"))

		try {
			await this.cli.search("health", this.workspacePath, { topK: 1 })
		} catch (error: any) {
			const downloadError = error?.message || String(error)

			if (downloadError.toLowerCase().includes("model") || downloadError.toLowerCase().includes("download")) {
				return {
					success: false,
					error: downloadError,
					wasRepaired: true,
				}
			}

			console.warn(`[SembleProvider] Smoke search returned error (may be non-model related): ${downloadError}`)
		}

		// Step 3: Verify model is now functional
		const finalCheck = await this.cli.checkModel()
		if (!finalCheck.installed) {
			const errMsg = finalCheck.error || "verification failed"
			if (errMsg.toLowerCase().includes("timeout") || errMsg.toLowerCase().includes("timed out")) {
				return {
					success: false,
					error: `Model downloaded but loading timed out. Please try again.`,
					wasRepaired: true,
				}
			}
			return {
				success: false,
				error: errMsg,
				wasRepaired: true,
			}
		}

		return { success: true, wasRepaired: true }
	}

	async startIndexing(): Promise<void> {
		if (!this._isInitialized) {
			await this.initialize()
		}

		if (this._state === "Error") {
			return
		}

		this._state = "Indexed"
		this.stateManager.setSystemState("Indexed", t("embeddings:semble.ready"))
	}

	stopIndexing(): void {
		// No-op: semble indexes on-the-fly per search call
	}

	async searchIndex(query: string, directoryPrefix?: string): Promise<VectorStoreSearchResult[]> {
		if (!this._isInitialized) {
			throw new Error("Semble provider is not initialized")
		}

		if (this._state === "Error") {
			const status = this.stateManager.getCurrentStatus()
			throw new Error(status.message || "Semble provider is in Error state")
		}

		try {
			return await this._executeSearch(query, directoryPrefix)
		} catch (error: any) {
			const errorMessage = error?.message || String(error)

			if (this._isModelCorruptionError(errorMessage)) {
				console.warn(`[SembleProvider] Search failed due to model issue, attempting repair: ${errorMessage}`)

				const repairResult = await this._repairModel()

				if (repairResult.success) {
					console.log("[SembleProvider] Model repaired successfully, retrying search...")
					return await this._executeSearch(query, directoryPrefix)
				} else {
					console.error(`[SembleProvider] Model repair failed: ${repairResult.error}`)
				}
			} else if (
				errorMessage.toLowerCase().includes("timeout") ||
				errorMessage.toLowerCase().includes("timed out")
			) {
				console.warn(`[SembleProvider] Search timed out (transient): ${errorMessage}`)
			}

			console.error("[SembleProvider] Search failed:", errorMessage)

			TelemetryService.instance.captureEvent(TelemetryEventName.CODE_INDEX_ERROR, {
				error: errorMessage,
				stack: error instanceof Error ? error.stack : undefined,
				location: "SembleProvider.searchIndex",
			})

			throw new Error(`Semble search failed: ${errorMessage}`)
		}
	}

	private async _executeSearch(query: string, directoryPrefix?: string): Promise<VectorStoreSearchResult[]> {
		console.log(`[SembleProvider] Searching in ${this.workspacePath}`)
		const results = await this.cli.search(query, this.workspacePath, {
			topK: this.config.topK,
			content: this.config.content,
		})

		let converted = this._convertResults(results, this.workspacePath)

		if (directoryPrefix) {
			const normalizedPrefix = path.join(this.workspacePath, directoryPrefix).replace(/\\/g, "/")
			converted = converted.filter((r) => {
				const filePath = (r.payload?.filePath ?? "").replace(/\\/g, "/")
				return filePath.startsWith(normalizedPrefix + "/") || filePath === normalizedPrefix
			})
			console.log(
				`[SembleProvider] Filtered to "${directoryPrefix}": ${converted.length} of ${results.length} results`,
			)
		}

		console.log(
			`[SembleProvider] Search returned ${converted.length} results (raw: ${results.length}). Sample path: ${converted[0]?.payload?.filePath ?? "none"}`,
		)
		return converted
	}

	private async _repairModel(): Promise<{ success: boolean; error?: string }> {
		try {
			this.stateManager.setSystemState("Indexing", t("embeddings:semble.repairingModel"))
			const clearResult = this.cli.clearModelCache()

			if (!clearResult.cleared) {
				return { success: false, error: clearResult.error || "Failed to clear model cache" }
			}

			this.stateManager.setSystemState("Indexing", t("embeddings:semble.reDownloadingModel"))
			await this.cli.search("health", this.workspacePath, { topK: 1 })

			const modelCheck = await this.cli.checkModel()
			if (!modelCheck.installed) {
				return { success: false, error: `Re-download completed but model not working: ${modelCheck.error}` }
			}

			return { success: true }
		} catch (error: any) {
			return { success: false, error: error?.message || String(error) }
		}
	}

	private _isModelCorruptionError(errorMessage: string): boolean {
		const lowerMsg = errorMessage.toLowerCase()

		if (lowerMsg.includes("timeout") || lowerMsg.includes("timed out")) {
			return false
		}

		const corruptionIndicators = [
			"model corrupted",
			"model files missing",
			"could not find expected model",
			"error while loading model",
			"model not found",
			"invalid model",
		]
		return corruptionIndicators.some((indicator) => lowerMsg.includes(indicator))
	}

	async clearIndexData(): Promise<void> {
		this._state = "Standby"
		this.stateManager.setSystemState("Standby", t("embeddings:semble.providerReset"))
	}

	dispose(): void {
		this._isInitialized = false
	}

	private _convertResults(results: SembleSearchResult[], basePath: string): VectorStoreSearchResult[] {
		const resolvedBase = path.resolve(basePath).replace(/\\/g, "/")

		const converted: VectorStoreSearchResult[] = []

		for (const [index, r] of results.entries()) {
			if (!r.chunk?.file_path) {
				continue
			}

			const filePath = path.join(basePath, r.chunk.file_path).replace(/\\/g, "/")

			const resolvedFilePath = path.resolve(basePath, r.chunk.file_path).replace(/\\/g, "/")

			if (!resolvedFilePath.startsWith(resolvedBase + "/") && resolvedFilePath !== resolvedBase) {
				continue
			}

			converted.push({
				id: `semble-${index}`,
				score: r.score,
				payload: {
					filePath,
					codeChunk: r.chunk?.content ?? "",
					startLine: r.chunk?.start_line ?? 0,
					endLine: r.chunk?.end_line ?? 0,
				},
			})
		}

		return converted
	}
}
