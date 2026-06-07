import { spawn } from "child_process"
import * as fs from "fs"
import * as path from "path"
import * as os from "os"

import { SembleSearchResult, SembleCheckResult, SembleContentType, SEMBLE_DEFAULTS } from "./types"

/**
 * HuggingFace cache directory for the potion-code-16M embedding model.
 * Semble downloads this model on first search via HuggingFace Hub.
 */
const HF_CACHE_MODEL_DIR = path.join(os.homedir(), ".cache", "huggingface", "hub", "models--minishlab--potion-code-16M")

/**
 * Expected model file patterns inside the HuggingFace cache snapshot directory.
 * Semble uses sentence-transformers with StaticEmbedding layout.
 * The snapshot contains model.safetensors (62MB weights) as a symlink to blobs/.
 */
const MODEL_FILE_PATTERNS = [
	"model.safetensors",
	"model.bin",
	"pytorch_model.bin",
	"0_StaticEmbedding/model.safetensors",
	"0_StaticEmbedding/model.bin",
]

/**
 * Wraps the `semble` CLI for programmatic access.
 *
 * The semble binary is automatically downloaded on enablement via semble-downloader.ts.
 *
 * All methods spawn the semble process via child_process.spawn with array
 * arguments (no shell) to prevent shell injection.
 *
 * Semble CLI (v0.3.0+) subcommands:
 *   search <query> [path]             — search a codebase
 *   find-related <file> <line> [path] — find similar code
 *   init                               — write sub-agent file
 *   savings                            — show token stats
 *
 * Common flags:
 *   -k, --top-k N                      — number of results (default: 5)
 *   --content TYPE [TYPE ...]          — content types: code, docs, config, all
 */
export class SembleCLI {
	private readonly semblePath: string

	constructor(semblePath: string) {
		this.semblePath = semblePath
	}

	/**
	 * Checks whether the semble binary is functional by running `semble --help`.
	 */
	async checkInstalled(): Promise<SembleCheckResult> {
		try {
			await this._spawn(["--help"], { timeout: 10_000 })
			return { installed: true }
		} catch (error: any) {
			return {
				installed: false,
				error: error?.stderr?.trim() || error?.message || "Failed to run semble",
			}
		}
	}

	/**
	 * Verifies the embedding model is available and intact.
	 *
	 * This performs a two-step check:
	 * 1. Verifies model files exist on disk (verifyModelFiles)
	 * 2. Runs a smoke search and checks stderr for corruption errors
	 *
	 * Semble downloads the potion-code-16M model from HuggingFace on first use.
	 * On first load, the model may take several minutes to download (62MB) and
	 * load into memory, so a generous timeout is used (15 minutes).
	 *
	 * If the download was interrupted, the cache may exist but contain incomplete
	 * or corrupted files. This method detects that case by checking both file
	 * presence AND the smoke search output.
	 */
	async checkModel(): Promise<SembleCheckResult> {
		// Step 1: Verify model files exist on disk
		const fileCheck = this.verifyModelFiles()
		if (!fileCheck.valid) {
			return {
				installed: false,
				error: fileCheck.error || "Model files are missing or corrupted",
			}
		}

		// Step 2: Run smoke search and check stderr for corruption errors
		// Even if exit code is 0, semble may report model corruption in stderr
		try {
			const { stderr } = await this._spawn(
				["search", "health", ".", "-k", "1"],
				{ timeout: 900_000 }, // 15 minutes for first model load
			)

			if (stderr && this._isModelCorruptionError(stderr)) {
				return {
					installed: false,
					error: `Model corrupted: ${stderr.trim()}`,
				}
			}

			return { installed: true }
		} catch (error: any) {
			const message = error?.message || String(error)

			// Process killed by timeout — model might still be loading
			if (message.includes("exceeded") || message.includes("timed out") || message.includes("killed")) {
				// Check stderr for actual corruption even on timeout
				const stderr = error?.stderr?.trim() || ""
				if (stderr && this._isModelCorruptionError(stderr)) {
					return {
						installed: false,
						error: `Model corrupted: ${stderr.trim()}`,
					}
				}
				// Otherwise timeout is not a corruption — might just need more time
				return {
					installed: false,
					error: `Model loading timed out (this is normal for first load). Will retry.`,
				}
			}

			const stderr = error?.stderr?.trim() || ""
			const stdout = error?.stdout?.trim() || ""
			return {
				installed: false,
				error: stderr || stdout || message,
			}
		}
	}

	/**
	 * Verifies that the potion-code-16M model files exist on disk and are not corrupted.
	 *
	 * Checks the HuggingFace cache directory for the model snapshot and looks for
	 * expected model weight files (model.safetensors, model.bin, etc.).
	 *
	 * A model is considered valid if at least one expected file exists and has
	 * a non-zero file size. The HuggingFace cache uses symlinks from snapshots/
	 * to blobs/, so a broken symlink (pointing to a deleted blob) will result
	 * in an ENOENT error, which we catch as "file not valid".
	 *
	 * @returns { valid: true } if model files are found, or { valid: false, error } if missing/corrupted.
	 */
	verifyModelFiles(): { valid: boolean; error?: string } {
		// Check if the model cache directory exists
		if (!fs.existsSync(HF_CACHE_MODEL_DIR)) {
			return { valid: false, error: "Model cache directory not found" }
		}

		// Find the snapshot directory
		const snapshotsDir = path.join(HF_CACHE_MODEL_DIR, "snapshots")
		if (!fs.existsSync(snapshotsDir)) {
			return { valid: false, error: "No snapshots directory found — model was never downloaded" }
		}

		const snapshotDirs = fs.readdirSync(snapshotsDir).filter((d) => {
			return fs.statSync(path.join(snapshotsDir, d)).isDirectory()
		})

		if (snapshotDirs.length === 0) {
			return { valid: false, error: "No snapshot directories found — model download may have failed" }
		}

		// Check snapshots from newest to oldest
		const sortedSnapshots = snapshotDirs.sort((a, b) => {
			const aTime = fs.statSync(path.join(snapshotsDir, a)).mtimeMs
			const bTime = fs.statSync(path.join(snapshotsDir, b)).mtimeMs
			return bTime - aTime
		})

		for (const snapshot of sortedSnapshots) {
			const snapshotPath = path.join(snapshotsDir, snapshot)

			// Check for expected model files in root of snapshot
			const found = this._checkSnapshotForFiles(snapshotPath)
			if (found) {
				return { valid: true }
			}
		}

		return {
			valid: false,
			error: `No valid model files found in any snapshot. Tried: ${MODEL_FILE_PATTERNS.join(", ")}`,
		}
	}

	/**
	 * Checks if any expected model file exists and has non-zero size in a snapshot directory.
	 * Handles symlinks properly (stat follows symlinks, fails with ENOENT if blob is missing).
	 */
	private _checkSnapshotForFiles(snapshotPath: string): boolean {
		for (const pattern of MODEL_FILE_PATTERNS) {
			try {
				const filePath = path.join(snapshotPath, pattern)
				const stats = fs.statSync(filePath) // follows symlinks, throws ENOENT if target missing
				if (stats.size > 0) {
					return true
				}
			} catch {
				// File doesn't exist or symlink is broken — try next pattern
				continue
			}
		}

		// Also check in subdirectories for sentence-transformers layout
		try {
			const entries = fs.readdirSync(snapshotPath)
			for (const entry of entries) {
				const entryPath = path.join(snapshotPath, entry)
				let isDir = false
				try {
					isDir = fs.statSync(entryPath).isDirectory()
				} catch {
					continue
				}
				if (!isDir) continue

				for (const pattern of MODEL_FILE_PATTERNS) {
					try {
						const filePath = path.join(entryPath, pattern)
						const stats = fs.statSync(filePath)
						if (stats.size > 0) {
							return true
						}
					} catch {
						continue
					}
				}
			}
		} catch {
			// Can't read directory — skip
		}

		return false
	}

	/**
	 * Clears the HuggingFace cache for the potion-code-16M model.
	 * This forces Semble to re-download the model on the next search.
	 *
	 * @returns { cleared: true } if cache was cleared, or { cleared: false, error } on failure.
	 */
	clearModelCache(): { cleared: boolean; error?: string } {
		try {
			if (fs.existsSync(HF_CACHE_MODEL_DIR)) {
				fs.rmSync(HF_CACHE_MODEL_DIR, { recursive: true, force: true })
				console.log(`[SembleCLI] Cleared model cache: ${HF_CACHE_MODEL_DIR}`)
			} else {
				console.log(`[SembleCLI] Model cache directory does not exist, nothing to clear`)
			}
			return { cleared: true }
		} catch (error: any) {
			return {
				cleared: false,
				error: `Failed to clear model cache: ${error?.message || String(error)}`,
			}
		}
	}

	/**
	 * Checks if a stderr string contains model corruption errors.
	 * Semble reports these even when exit code is 0.
	 */
	private _isModelCorruptionError(stderr: string): boolean {
		const corruptionIndicators = [
			"Could not find expected model files",
			"Error while loading model",
			"Model not found",
			"Invalid model",
			"corrupted",
			"model2vec",
			"sentence-transformers",
			"0_StaticEmbedding",
		]
		const lowerStderr = stderr.toLowerCase()
		return corruptionIndicators.some((indicator) => lowerStderr.includes(indicator.toLowerCase()))
	}

	/**
	 * Searches a codebase. Semble indexes on-the-fly during search.
	 *
	 * Usage: semble search <query> [path] [-k N] [--content TYPE [TYPE ...]]
	 *
	 * On first search, semble downloads and loads the embedding model (62MB),
	 * which may take several minutes. Uses a 15-minute timeout.
	 */
	async search(
		query: string,
		repoPath: string,
		options?: { topK?: number; content?: SembleContentType },
	): Promise<SembleSearchResult[]> {
		const topK = options?.topK ?? SEMBLE_DEFAULTS.DEFAULT_TOP_K
		const args = ["search", query, repoPath, "-k", String(topK)]
		if (options?.content && options.content !== "code") {
			args.push("--content", options.content)
		}

		try {
			const { stdout, stderr } = await this._spawn(args, { timeout: 900_000 }) // 15 minutes

			// Check stderr for model corruption even on successful exit
			if (stderr && this._isModelCorruptionError(stderr)) {
				throw new Error(`Model corrupted: ${stderr.trim()}`)
			}

			return this._parseOutput(stdout)
		} catch (error: any) {
			const stderr = error?.stderr?.trim() || ""
			const message = error?.message || String(error)
			throw new Error(`Semble search failed: ${stderr || message}`)
		}
	}

	/**
	 * Finds code similar to a known location.
	 *
	 * Usage: semble find-related <file_path> <line> [path] [-k N] [--content TYPE [TYPE ...]]
	 */
	async findRelated(
		filePath: string,
		line: number,
		repoPath: string,
		options?: { topK?: number; content?: SembleContentType },
	): Promise<SembleSearchResult[]> {
		const topK = options?.topK ?? SEMBLE_DEFAULTS.DEFAULT_TOP_K
		const args = ["find-related", filePath, String(line), repoPath, "-k", String(topK)]
		if (options?.content && options.content !== "code") {
			args.push("--content", options.content)
		}

		try {
			const { stdout } = await this._spawn(args, { timeout: 900_000 })
			return this._parseOutput(stdout)
		} catch (error: any) {
			const stderr = error?.stderr?.trim() || ""
			const message = error?.message || String(error)
			throw new Error(`Semble find-related failed: ${stderr || message}`)
		}
	}

	/**
	 * Spawns the semble process and collects stdout/stderr.
	 * Uses spawn without shell — args are passed as an array, no injection risk.
	 * Caps stdout/stderr buffers at MAX_BUFFER_BYTES to prevent OOM in the extension host.
	 * Kills the process and rejects if the cap is exceeded.
	 */
	private _spawn(args: string[], options: { timeout: number }): Promise<{ stdout: string; stderr: string }> {
		const MAX_BUFFER_BYTES = 10 * 1024 * 1024 // 10 MB

		return new Promise((resolve, reject) => {
			const child = spawn(this.semblePath, args, {
				shell: false,
				timeout: options.timeout,
				stdio: ["ignore", "pipe", "pipe"],
			})

			let stdout = ""
			let stderr = ""
			let stdoutBytes = 0
			let stderrBytes = 0
			let killed = false

			child.stdout?.on("data", (data: Buffer) => {
				stdoutBytes += data.length
				if (stdoutBytes <= MAX_BUFFER_BYTES) {
					stdout += data.toString()
				} else if (!killed) {
					killed = true
					child.kill()
					reject({
						message: `stdout exceeded ${MAX_BUFFER_BYTES} bytes — process killed to protect extension host`,
						stderr,
					})
				}
			})

			child.stderr?.on("data", (data: Buffer) => {
				stderrBytes += data.length
				if (stderrBytes <= MAX_BUFFER_BYTES) {
					stderr += data.toString()
				} else if (!killed) {
					killed = true
					child.kill()
					reject({
						message: `stderr exceeded ${MAX_BUFFER_BYTES} bytes — process killed to protect extension host`,
						stderr,
					})
				}
			})

			child.on("error", (err: Error) => {
				if (!killed) {
					reject({ message: err.message, stderr })
				}
			})

			child.on("close", (code: number | null) => {
				if (killed) {
					return // already rejected
				}
				if (code === 0) {
					resolve({ stdout, stderr })
				} else {
					reject({ message: `Process exited with code ${code}`, stderr, stdout })
				}
			})
		})
	}

	/**
	 * Parses semble CLI JSON output into structured results.
	 *
	 * Semble v0.3.0+ outputs JSON by default with format:
	 *   { "query": "...", "results": [{ "chunk": { "content": "...", "file_path": "...", "start_line": N, "end_line": M, "language": "...", "location": "..." }, "score": X }] }
	 *
	 * If the query returns no results, semble outputs:
	 *   { "error": "No results found." }
	 */
	private _parseOutput(stdout: string): SembleSearchResult[] {
		const trimmed = stdout.trim()
		if (!trimmed) {
			return []
		}

		try {
			const parsed = JSON.parse(trimmed)

			// Handle error response: {"error": "No results found."}
			if (parsed.error) {
				return []
			}

			// Handle successful response: {query, results: [{chunk, score}]}
			if (parsed.results && Array.isArray(parsed.results)) {
				return parsed.results as SembleSearchResult[]
			}

			// Fallback: if it's a flat array (older format)
			if (Array.isArray(parsed)) {
				return parsed as SembleSearchResult[]
			}

			return []
		} catch {
			// Not JSON — this shouldn't happen with v0.3.0+ but handle gracefully
			console.warn("[SembleCLI] Unexpected non-JSON output from semble")
			return []
		}
	}
}
