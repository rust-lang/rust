import * as cp from "child_process";
import * as os from "os";
import * as path from "path";
import * as readline from "readline";
import * as vscode from "vscode";
import { execute, log, memoizeAsync } from "./util";

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

export class Cargo {
    constructor(readonly rootFolder: string, readonly output: vscode.OutputChannel) {}

    // Made public for testing purposes
    static artifactSpec(args: readonly string[]): ArtifactSpec {
        const cargoArgs = [...args, "--message-format=json"];

        // arguments for a runnable from the quick pick should be updated.
        // see crates\rust-analyzer\src\main_loop\handlers.rs, handle_code_lens
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

        return result;
    }

    private async getArtifacts(spec: ArtifactSpec): Promise<CompilationArtifact[]> {
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
                        this.output.append(message.message.rendered);
                    }
                },
                (stderr) => this.output.append(stderr)
            );
        } catch (err) {
            this.output.show(true);
            throw new Error(`Cargo invocation has failed: ${err}`);
        }

        return spec.filter?.(artifacts) ?? artifacts;
    }

    async executableFromArgs(args: readonly string[]): Promise<string> {
        const artifacts = await this.getArtifacts(Cargo.artifactSpec(args));

        if (artifacts.length === 0) {
            throw new Error("No compilation artifacts");
        } else if (artifacts.length > 1) {
            throw new Error("Multiple compilation artifacts are not supported.");
        }

        return artifacts[0].fileName;
    }

    private async runCargo(
        cargoArgs: string[],
        onStdoutJson: (obj: any) => void,
        onStderrString: (data: string) => void
    ): Promise<number> {
        const path = await cargoPath();
        return await new Promise((resolve, reject) => {
            const cargo = cp.spawn(path, cargoArgs, {
                stdio: ["ignore", "pipe", "pipe"],
                cwd: this.rootFolder,
            });

            cargo.on("error", (err) => reject(new Error(`could not launch cargo: ${err}`)));

            cargo.stderr.on("data", (chunk) => onStderrString(chunk.toString()));

            const rl = readline.createInterface({ input: cargo.stdout });
            rl.on("line", (line) => {
                const message = JSON.parse(line);
                onStdoutJson(message);
            });

            cargo.on("exit", (exitCode, _) => {
                if (exitCode === 0) resolve(exitCode);
                else reject(new Error(`exit code: ${exitCode}.`));
            });
        });
    }
}

/** Mirrors `project_model::sysroot::discover_sysroot_dir()` implementation*/
export async function getSysroot(dir: string): Promise<string> {
    const rustcPath = await getPathForExecutable("rustc");

    // do not memoize the result because the toolchain may change between runs
    return await execute(`${rustcPath} --print sysroot`, { cwd: dir });
}

export async function getRustcId(dir: string): Promise<string> {
    const rustcPath = await getPathForExecutable("rustc");

    // do not memoize the result because the toolchain may change between runs
    const data = await execute(`${rustcPath} -V -v`, { cwd: dir });
    const rx = /commit-hash:\s(.*)$/m;

    return rx.exec(data)![1];
}

/** Mirrors `toolchain::cargo()` implementation */
export function cargoPath(): Promise<string> {
    return getPathForExecutable("cargo");
}

/** Mirrors `toolchain::get_path_for_executable()` implementation */
export const getPathForExecutable = memoizeAsync(
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
    }
);

async function lookupInPath(exec: string): Promise<boolean> {
    const paths = process.env.PATH ?? "";

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
