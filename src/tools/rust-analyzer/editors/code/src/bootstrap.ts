import * as vscode from "vscode";
import * as os from "os";
import type { Config } from "./config";
import { type Env, log, spawnAsync } from "./util";
import type { PersistentState } from "./persistent_state";
import { exec } from "child_process";
import { TextDecoder } from "node:util";

export async function bootstrap(
    context: vscode.ExtensionContext,
    config: Config,
    state: PersistentState,
): Promise<string> {
    const path = await getServer(context, config, state);
    if (!path) {
        throw new Error(
            "rust-analyzer Language Server is not available. " +
                "Please, ensure its [proper installation](https://rust-analyzer.github.io/book/installation.html).",
        );
    }

    log.info("Using server binary at", path);

    if (!isValidExecutable(path, config.serverExtraEnv)) {
        throw new Error(
            `Failed to execute ${path} --version.` +
                (config.serverPath
                    ? `\`config.server.path\` or \`config.serverPath\` has been set explicitly.\
            Consider removing this config or making a valid server binary available at that path.`
                    : ""),
        );
    }

    return path;
}
async function getServer(
    context: vscode.ExtensionContext,
    config: Config,
    state: PersistentState,
): Promise<string | undefined> {
    const packageJson: {
        version: string;
        releaseTag: string | null;
        enableProposedApi: boolean | undefined;
    } = context.extension.packageJSON;

    // check if the server path is configured explicitly
    const explicitPath = process.env["__RA_LSP_SERVER_DEBUG"] ?? config.serverPath;
    if (explicitPath) {
        if (explicitPath.startsWith("~/")) {
            return os.homedir() + explicitPath.slice("~".length);
        }
        return explicitPath;
    }

    let toolchainServerPath = undefined;
    if (vscode.workspace.workspaceFolders) {
        for (const workspaceFolder of vscode.workspace.workspaceFolders) {
            // otherwise check if there is a toolchain override for the current vscode workspace
            // and if the toolchain of this override has a rust-analyzer component
            // if so, use the rust-analyzer component
            const toolchainUri = vscode.Uri.joinPath(workspaceFolder.uri, "rust-toolchain.toml");
            if (await hasToolchainFileWithRaDeclared(toolchainUri)) {
                const res = await spawnAsync("rustup", ["which", "rust-analyzer"], {
                    env: { ...process.env },
                    cwd: workspaceFolder.uri.fsPath,
                });
                if (!res.error && res.status === 0) {
                    toolchainServerPath = await earliestToolchainPath(
                        toolchainServerPath,
                        res.stdout.trim(),
                        raVersionResolver,
                    );
                }
            }
        }
    }
    if (toolchainServerPath) {
        return toolchainServerPath;
    }

    if (packageJson.releaseTag === null) return "rust-analyzer";

    // finally, use the bundled one
    const ext = process.platform === "win32" ? ".exe" : "";
    const bundled = vscode.Uri.joinPath(context.extensionUri, "server", `rust-analyzer${ext}`);
    const bundledExists = await fileExists(bundled);
    if (bundledExists) {
        let server = bundled;
        if (await isNixOs()) {
            server = await getNixOsServer(
                context.globalStorageUri,
                packageJson.version,
                ext,
                state,
                bundled,
                server,
            );
            await state.updateServerVersion(packageJson.version);
        }
        return server.fsPath;
    }

    await vscode.window.showErrorMessage(
        "Unfortunately we don't ship binaries for your platform yet. " +
            "You need to manually clone the rust-analyzer repository and " +
            "run `cargo xtask install --server` to build the language server from sources. " +
            "If you feel that your platform should be supported, please create an issue " +
            "about that [here](https://github.com/rust-lang/rust-analyzer/issues) and we " +
            "will consider it.",
    );
    return undefined;
}

// Given a path to a rust-analyzer executable, resolve its version and return it.
async function raVersionResolver(path: string): Promise<string | undefined> {
    const res = await spawnAsync(path, ["--version"]);
    if (!res.error && res.status === 0) {
        return res.stdout;
    } else {
        return undefined;
    }
}

// Given a path to two rust-analyzer executables, return the earliest one by date.
async function earliestToolchainPath(
    path0: string | undefined,
    path1: string,
    raVersionResolver: (path: string) => Promise<string | undefined>,
): Promise<string> {
    if (path0) {
        if (
            (await orderFromPath(path0, raVersionResolver)) <
            (await orderFromPath(path1, raVersionResolver))
        ) {
            return path0;
        } else {
            return path1;
        }
    } else {
        return path1;
    }
}

// Further to extracting a date for comparison, determine the order of a toolchain as follows:
//  Highest - nightly
//  Medium  - versioned
//  Lowest  - stable
// Example paths:
//  nightly   - /Users/myuser/.rustup/toolchains/nightly-2022-11-22-aarch64-apple-darwin/bin/rust-analyzer
//  versioned - /Users/myuser/.rustup/toolchains/1.72.1-aarch64-apple-darwin/bin/rust-analyzer
//  stable    - /Users/myuser/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rust-analyzer
async function orderFromPath(
    path: string,
    raVersionResolver: (path: string) => Promise<string | undefined>,
): Promise<string> {
    const raVersion = await raVersionResolver(path);
    const raDate = raVersion?.match(/^rust-analyzer .*\(.* (\d{4}-\d{2}-\d{2})\)$/);
    if (raDate?.length === 2) {
        const precedence = path.includes("nightly-") ? "0" : "1";
        return "0-" + raDate[1] + "/" + precedence;
    } else {
        return "2";
    }
}

async function fileExists(uri: vscode.Uri) {
    return await vscode.workspace.fs.stat(uri).then(
        () => true,
        () => false,
    );
}

async function hasToolchainFileWithRaDeclared(uri: vscode.Uri): Promise<boolean> {
    try {
        const toolchainFileContents = new TextDecoder().decode(
            await vscode.workspace.fs.readFile(uri),
        );
        return (
            toolchainFileContents.match(/components\s*=\s*\[.*"rust-analyzer".*\]/g)?.length === 1
        );
    } catch (_) {
        return false;
    }
}

export async function isValidExecutable(path: string, extraEnv: Env): Promise<boolean> {
    log.debug("Checking availability of a binary at", path);

    const newEnv = { ...process.env };
    for (const [k, v] of Object.entries(extraEnv)) {
        if (v) {
            newEnv[k] = v;
        } else if (k in newEnv) {
            delete newEnv[k];
        }
    }
    const res = await spawnAsync(path, ["--version"], {
        env: newEnv,
    });

    if (res.error) {
        log.warn(path, "--version:", res);
    } else {
        log.info(path, "--version:", res);
    }
    return res.status === 0;
}

async function getNixOsServer(
    globalStorageUri: vscode.Uri,
    version: string,
    ext: string,
    state: PersistentState,
    bundled: vscode.Uri,
    server: vscode.Uri,
) {
    await vscode.workspace.fs.createDirectory(globalStorageUri).then();
    const dest = vscode.Uri.joinPath(globalStorageUri, `rust-analyzer${ext}`);
    let exists = await vscode.workspace.fs.stat(dest).then(
        () => true,
        () => false,
    );
    if (exists && version !== state.serverVersion) {
        await vscode.workspace.fs.delete(dest);
        exists = false;
    }
    if (!exists) {
        await vscode.workspace.fs.copy(bundled, dest);
        await patchelf(dest);
    }
    server = dest;
    return server;
}

async function isNixOs(): Promise<boolean> {
    try {
        const contents = (
            await vscode.workspace.fs.readFile(vscode.Uri.file("/etc/os-release"))
        ).toString();
        const idString = contents.split("\n").find((a) => a.startsWith("ID=")) || "ID=linux";
        return idString.indexOf("nixos") !== -1;
    } catch {
        return false;
    }
}

async function patchelf(dest: vscode.Uri): Promise<void> {
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: "Patching rust-analyzer for NixOS",
        },
        async (progress, _) => {
            const expression = `
            {srcStr, pkgs ? import <nixpkgs> {}}:
                pkgs.stdenv.mkDerivation {
                    name = "rust-analyzer";
                    src = /. + srcStr;
                    phases = [ "installPhase" "fixupPhase" ];
                    installPhase = "cp $src $out";
                    fixupPhase = ''
                    chmod 755 $out
                    patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" $out
                    '';
                }
            `;
            const origFile = vscode.Uri.file(dest.fsPath + "-orig");
            await vscode.workspace.fs.rename(dest, origFile, { overwrite: true });
            try {
                progress.report({ message: "Patching executable", increment: 20 });
                await new Promise((resolve, reject) => {
                    const handle = exec(
                        `nix-build -E - --argstr srcStr '${origFile.fsPath}' -o '${dest.fsPath}'`,
                        (err, stdout, stderr) => {
                            if (err != null) {
                                reject(Error(stderr));
                            } else {
                                resolve(stdout);
                            }
                        },
                    );
                    handle.stdin?.write(expression);
                    handle.stdin?.end();
                });
            } finally {
                await vscode.workspace.fs.delete(origFile);
            }
        },
    );
}

export const _private = {
    earliestToolchainPath,
    orderFromPath,
};
