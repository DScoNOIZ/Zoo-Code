import { spawn, spawnSync } from "child_process"

import { SembleSearchResult, SembleCheckResult, SembleContentType, SEMBLE_DEFAULTS } from "./types"

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
	 * Verifies the embedding model is available by running a smoke search.
	 *
	 * Semble downloads the potion-code-16M model from HuggingFace on first use.
	 * On first load, the model may take several minutes to download (62MB) and
	 * load into memory, so a generous timeout is used (15 minutes).
	 *
	 * This method uses only the CLI surface — it does not inspect internal
	 * HuggingFace cache directory structure, which may change between versions.
	 */
	async checkModel(): Promise<SembleCheckResult> {
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
				const stderr = error?.stderr?.trim() || ""
				if (stderr && this._isModelCorruptionError(stderr)) {
					return {
						installed: false,
						error: `Model corrupted: ${stderr.trim()}`,
					}
				}
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
	 * Clears the HuggingFace cache for the potion-code-16M model
	 * by removing Semble's model cache directory.
	 *
	 * Uses `semble clear-cache` CLI command when available, otherwise
	 * falls back to removing the known HuggingFace cache path.
	 *
	 * @returns { cleared: true } if cache was cleared, or { cleared: false, error } on failure.
	 */
	clearModelCache(): { cleared: boolean; error?: string } {
		try {
			// Use CLI surface to clear cache — avoids hardcoding HF paths
			this._spawnSync(["clear-cache"], { timeout: 30_000 })
			console.log("[SembleCLI] Model cache cleared via CLI")
			return { cleared: true }
		} catch {
			// Fallback: clear-cache command may not exist in older versions
			return { cleared: false, error: "clear-cache command not available" }
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
	 * Synchronous spawn for short-running commands (e.g., clear-cache).
	 */
	private _spawnSync(args: string[], options: { timeout: number }): void {
		const result = spawnSync(this.semblePath, args, {
			shell: false,
			timeout: options.timeout,
			stdio: ["ignore", "pipe", "pipe"],
		})

		if (result.error) {
			throw result.error
		}

		if (result.status !== 0) {
			const stderr = (result.stderr?.toString() || "").trim()
			const stdout = (result.stdout?.toString() || "").trim()
			throw new Error(stderr || stdout || `Process exited with code ${result.status}`)
		}
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
