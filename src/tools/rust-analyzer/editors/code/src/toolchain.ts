import * as cp from "child_process";
import * as os from "os";
import * as path from "path";
import * as readline from "readline";
import * as vscode from "vscode";
import { Env, log, memoizeAsync, unwrapUndefinable } from "./util";
import type { CargoRunnableArgs } from "./lsp_ext";

interface CompilationArtifact {
    fileName: string;
    name: string;
    kind: string;
    isTest: boolean;
}

export interface ArtifactSpec {
    cargoArgs: string[];
    filter?: (artifacts: CompilationArtifact[]) => CompilationArtifact[];
}

interface CompilerMessage {
    reason: string;
    executable?: string;
    target: {
        crate_types: [string, ...string[]];
        kind: [string, ...string[]];
        name: string;
    };
    profile: {
        test: boolean;
    };
    message: {
        rendered: string;
    };
}

export class Cargo {
    constructor(
        readonly rootFolder: string,
        readonly env: Env,
    ) {}

    // Made public for testing purposes
    static artifactSpec(cargoArgs: string[], executableArgs?: string[]): ArtifactSpec {
        cargoArgs = [...cargoArgs, "--message-format=json"];
        // arguments for a runnable from the quick pick should be updated.
        // see crates\rust-analyzer\src\handlers\request.rs, handle_code_lens
        switch (cargoArgs[0]) {
            case "run":
                cargoArgs[0] = "build";
                break;
            case "test": {
                if (!cargoArgs.includes("--no-run")) {
                    cargoArgs.push("--no-run");
                }
                break;
            }
        }

        const result: ArtifactSpec = { cargoArgs: cargoArgs };
        if (cargoArgs[0] === "test" || cargoArgs[0] === "bench") {
            // for instance, `crates\rust-analyzer\tests\heavy_tests\main.rs` tests
            // produce 2 artifacts: {"kind": "bin"} and {"kind": "test"}
            result.filter = (artifacts) => artifacts.filter((it) => it.isTest);
        }
        if (executableArgs) {
            cargoArgs.push("--", ...executableArgs);
        }

        return result;
    }

    private async getArtifacts(
        spec: ArtifactSpec,
        env?: Record<string, string>,
    ): Promise<CompilationArtifact[]> {
        const artifacts: CompilationArtifact[] = [];

        try {
            await this.runCargo(
                spec.cargoArgs,
                (message) => {
                    if (message.reason === "compiler-artifact" && message.executable) {
                        const isBinary = message.target.crate_types.includes("bin");
                        const isBuildScript = message.target.kind.includes("custom-build");
                        if ((isBinary && !isBuildScript) || message.profile.test) {
                            artifacts.push({
                                fileName: message.executable,
                                name: message.target.name,
                                kind: message.target.kind[0],
                                isTest: message.profile.test,
                            });
                        }
                    } else if (message.reason === "compiler-message") {
                        log.info(message.message.rendered);
                    }
                },
                (stderr) => log.error(stderr),
                env,
            );
        } catch (err) {
            log.error(`Cargo invocation has failed: ${err}`);
            throw new Error(`Cargo invocation has failed: ${err}`);
        }

        return spec.filter?.(artifacts) ?? artifacts;
    }

    async executableFromArgs(runnableArgs: CargoRunnableArgs): Promise<string> {
        const artifacts = await this.getArtifacts(
            Cargo.artifactSpec(runnableArgs.cargoArgs, runnableArgs.executableArgs),
            runnableArgs.environment,
        );

        if (artifacts.length === 0) {
            throw new Error("No compilation artifacts");
        } else if (artifacts.length > 1) {
            throw new Error("Multiple compilation artifacts are not supported.");
        }

        const artifact = unwrapUndefinable(artifacts[0]);
        return artifact.fileName;
    }

    private async runCargo(
        cargoArgs: string[],
        onStdoutJson: (obj: CompilerMessage) => void,
        onStderrString: (data: string) => void,
        env?: Record<string, string>,
    ): Promise<number> {
        const path = await cargoPath(env);
        return await new Promise((resolve, reject) => {
            const cargo = cp.spawn(path, cargoArgs, {
                stdio: ["ignore", "pipe", "pipe"],
                cwd: this.rootFolder,
                env: this.env,
            });

            cargo.on("error", (err) => reject(new Error(`could not launch cargo: ${err}`)));

            cargo.stderr.on("data", (chunk) => onStderrString(chunk.toString()));

            const rl = readline.createInterface({ input: cargo.stdout });
            rl.on("line", (line) => {
                const message = JSON.parse(line);
                onStdoutJson(message);
            });

            cargo.on("exit", (exitCode) => {
                if (exitCode === 0) resolve(exitCode);
                else reject(new Error(`exit code: ${exitCode}.`));
            });
        });
    }
}

/** Mirrors `toolchain::cargo()` implementation */
// FIXME: The server should provide this
export function cargoPath(env?: Env): Promise<string> {
    if (env?.["RUSTC_TOOLCHAIN"]) {
        return Promise.resolve("cargo");
    }
    return getPathForExecutable("cargo");
}

/** Mirrors `toolchain::get_path_for_executable()` implementation */
const getPathForExecutable = memoizeAsync(
    // We apply caching to decrease file-system interactions
    async (executableName: "cargo" | "rustc" | "rustup"): Promise<string> => {
        {
            const envVar = process.env[executableName.toUpperCase()];
            if (envVar) return envVar;
        }

        if (await lookupInPath(executableName)) return executableName;

        const cargoHome = getCargoHome();
        if (cargoHome) {
            const standardPath = vscode.Uri.joinPath(cargoHome, "bin", executableName);
            if (await isFileAtUri(standardPath)) return standardPath.fsPath;
        }
        return executableName;
    },
);

async function lookupInPath(exec: string): Promise<boolean> {
    const paths = process.env["PATH"] ?? "";

    const candidates = paths.split(path.delimiter).flatMap((dirInPath) => {
        const candidate = path.join(dirInPath, exec);
        return os.type() === "Windows_NT" ? [candidate, `${candidate}.exe`] : [candidate];
    });

    for await (const isFile of candidates.map(isFileAtPath)) {
        if (isFile) {
            return true;
        }
    }
    return false;
}

function getCargoHome(): vscode.Uri | null {
    const envVar = process.env["CARGO_HOME"];
    if (envVar) return vscode.Uri.file(envVar);

    try {
        // hmm, `os.homedir()` seems to be infallible
        // it is not mentioned in docs and cannot be inferred by the type signature...
        return vscode.Uri.joinPath(vscode.Uri.file(os.homedir()), ".cargo");
    } catch (err) {
        log.error("Failed to read the fs info", err);
    }

    return null;
}

async function isFileAtPath(path: string): Promise<boolean> {
    return isFileAtUri(vscode.Uri.file(path));
}

async function isFileAtUri(uri: vscode.Uri): Promise<boolean> {
    try {
        return ((await vscode.workspace.fs.stat(uri)).type & vscode.FileType.File) !== 0;
    } catch {
        return false;
    }
}
