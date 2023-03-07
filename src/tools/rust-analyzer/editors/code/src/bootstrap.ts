import * as vscode from "vscode";
import * as os from "os";
import { Config } from "./config";
import { log, isValidExecutable } from "./util";
import { PersistentState } from "./persistent_state";
import { exec } from "child_process";

export async function bootstrap(
    context: vscode.ExtensionContext,
    config: Config,
    state: PersistentState
): Promise<string> {
    const path = await getServer(context, config, state);
    if (!path) {
        throw new Error(
            "Rust Analyzer Language Server is not available. " +
                "Please, ensure its [proper installation](https://rust-analyzer.github.io/manual.html#installation)."
        );
    }

    log.info("Using server binary at", path);

    if (!isValidExecutable(path)) {
        if (config.serverPath) {
            throw new Error(`Failed to execute ${path} --version. \`config.server.path\` or \`config.serverPath\` has been set explicitly.\
            Consider removing this config or making a valid server binary available at that path.`);
        } else {
            throw new Error(`Failed to execute ${path} --version`);
        }
    }

    return path;
}
async function getServer(
    context: vscode.ExtensionContext,
    config: Config,
    state: PersistentState
): Promise<string | undefined> {
    const explicitPath = process.env.__RA_LSP_SERVER_DEBUG ?? config.serverPath;
    if (explicitPath) {
        if (explicitPath.startsWith("~/")) {
            return os.homedir() + explicitPath.slice("~".length);
        }
        return explicitPath;
    }
    if (config.package.releaseTag === null) return "rust-analyzer";

    const ext = process.platform === "win32" ? ".exe" : "";
    const bundled = vscode.Uri.joinPath(context.extensionUri, "server", `rust-analyzer${ext}`);
    const bundledExists = await vscode.workspace.fs.stat(bundled).then(
        () => true,
        () => false
    );
    if (bundledExists) {
        let server = bundled;
        if (await isNixOs()) {
            await vscode.workspace.fs.createDirectory(config.globalStorageUri).then();
            const dest = vscode.Uri.joinPath(config.globalStorageUri, `rust-analyzer${ext}`);
            let exists = await vscode.workspace.fs.stat(dest).then(
                () => true,
                () => false
            );
            if (exists && config.package.version !== state.serverVersion) {
                await vscode.workspace.fs.delete(dest);
                exists = false;
            }
            if (!exists) {
                await vscode.workspace.fs.copy(bundled, dest);
                await patchelf(dest);
            }
            server = dest;
        }
        await state.updateServerVersion(config.package.version);
        return server.fsPath;
    }

    await state.updateServerVersion(undefined);
    await vscode.window.showErrorMessage(
        "Unfortunately we don't ship binaries for your platform yet. " +
            "You need to manually clone the rust-analyzer repository and " +
            "run `cargo xtask install --server` to build the language server from sources. " +
            "If you feel that your platform should be supported, please create an issue " +
            "about that [here](https://github.com/rust-lang/rust-analyzer/issues) and we " +
            "will consider it."
    );
    return undefined;
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
                        }
                    );
                    handle.stdin?.write(expression);
                    handle.stdin?.end();
                });
            } finally {
                await vscode.workspace.fs.delete(origFile);
            }
        }
    );
}
