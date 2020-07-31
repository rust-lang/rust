import * as vscode from 'vscode';
import * as path from "path";
import * as os from "os";
import { promises as fs, PathLike } from "fs";

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { Ctx } from './ctx';
import { Config, NIGHTLY_TAG } from './config';
import { log, assert, isValidExecutable } from './util';
import { PersistentState } from './persistent_state';
import { fetchRelease, download } from './net';
import { activateTaskProvider } from './tasks';
import { setContextValue } from './util';
import { exec } from 'child_process';

let ctx: Ctx | undefined;

const RUST_PROJECT_CONTEXT_NAME = "inRustProject";

export async function activate(context: vscode.ExtensionContext) {
    // For some reason vscode not always shows pop-up error notifications
    // when an extension fails to activate, so we do it explicitly by ourselves.
    // FIXME: remove this bit of code once vscode fixes this issue: https://github.com/microsoft/vscode/issues/101242
    await tryActivate(context).catch(err => {
        void vscode.window.showErrorMessage(`Cannot activate rust-analyzer: ${err.message}`);
        throw err;
    });
}

async function tryActivate(context: vscode.ExtensionContext) {
    // Register a "dumb" onEnter command for the case where server fails to
    // start.
    //
    // FIXME: refactor command registration code such that commands are
    // **always** registered, even if the server does not start. Use API like
    // this perhaps?
    //
    // ```TypeScript
    // registerCommand(
    //    factory: (Ctx) => ((Ctx) => any),
    //    fallback: () => any = () => vscode.window.showErrorMessage(
    //        "rust-analyzer is not available"
    //    ),
    // )
    const defaultOnEnter = vscode.commands.registerCommand(
        'rust-analyzer.onEnter',
        () => vscode.commands.executeCommand('default:type', { text: '\n' }),
    );
    context.subscriptions.push(defaultOnEnter);

    const config = new Config(context);
    const state = new PersistentState(context.globalState);
    const serverPath = await bootstrap(config, state).catch(err => {
        let message = "bootstrap error. ";

        if (err.code === "EBUSY" || err.code === "ETXTBSY" || err.code === "EPERM") {
            message += "Other vscode windows might be using rust-analyzer, ";
            message += "you should close them and reload this window to retry. ";
        }

        message += 'See the logs in "OUTPUT > Rust Analyzer Client" (should open automatically). ';
        message += 'To enable verbose logs use { "rust-analyzer.trace.extension": true }';

        log.error("Bootstrap error", err);
        throw new Error(message);
    });

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (workspaceFolder === undefined) {
        throw new Error("no folder is opened");
    }

    // Note: we try to start the server before we activate type hints so that it
    // registers its `onDidChangeDocument` handler before us.
    //
    // This a horribly, horribly wrong way to deal with this problem.
    ctx = await Ctx.create(config, context, serverPath, workspaceFolder.uri.fsPath);

    setContextValue(RUST_PROJECT_CONTEXT_NAME, true);

    // Commands which invokes manually via command palette, shortcut, etc.

    // Reloading is inspired by @DanTup maneuver: https://github.com/microsoft/vscode/issues/45774#issuecomment-373423895
    ctx.registerCommand('reload', _ => async () => {
        void vscode.window.showInformationMessage('Reloading rust-analyzer...');
        await deactivate();
        while (context.subscriptions.length > 0) {
            try {
                context.subscriptions.pop()!.dispose();
            } catch (err) {
                log.error("Dispose error:", err);
            }
        }
        await activate(context).catch(log.error);
    });

    ctx.registerCommand('analyzerStatus', commands.analyzerStatus);
    ctx.registerCommand('memoryUsage', commands.memoryUsage);
    ctx.registerCommand('reloadWorkspace', commands.reloadWorkspace);
    ctx.registerCommand('matchingBrace', commands.matchingBrace);
    ctx.registerCommand('joinLines', commands.joinLines);
    ctx.registerCommand('parentModule', commands.parentModule);
    ctx.registerCommand('syntaxTree', commands.syntaxTree);
    ctx.registerCommand('expandMacro', commands.expandMacro);
    ctx.registerCommand('run', commands.run);
    ctx.registerCommand('debug', commands.debug);
    ctx.registerCommand('newDebugConfig', commands.newDebugConfig);

    defaultOnEnter.dispose();
    ctx.registerCommand('onEnter', commands.onEnter);

    ctx.registerCommand('ssr', commands.ssr);
    ctx.registerCommand('serverVersion', commands.serverVersion);
    ctx.registerCommand('toggleInlayHints', commands.toggleInlayHints);

    // Internal commands which are invoked by the server.
    ctx.registerCommand('runSingle', commands.runSingle);
    ctx.registerCommand('debugSingle', commands.debugSingle);
    ctx.registerCommand('showReferences', commands.showReferences);
    ctx.registerCommand('applySnippetWorkspaceEdit', commands.applySnippetWorkspaceEditCommand);
    ctx.registerCommand('resolveCodeAction', commands.resolveCodeAction);
    ctx.registerCommand('applyActionGroup', commands.applyActionGroup);
    ctx.registerCommand('gotoLocation', commands.gotoLocation);

    ctx.pushCleanup(activateTaskProvider(workspaceFolder, ctx.config));

    activateInlayHints(ctx);

    vscode.workspace.onDidChangeConfiguration(
        _ => ctx?.client?.sendNotification('workspace/didChangeConfiguration', { settings: "" }),
        null,
        ctx.subscriptions,
    );
}

export async function deactivate() {
    setContextValue(RUST_PROJECT_CONTEXT_NAME, undefined);
    await ctx?.client.stop();
    ctx = undefined;
}

async function bootstrap(config: Config, state: PersistentState): Promise<string> {
    await fs.mkdir(config.globalStoragePath, { recursive: true });

    await bootstrapExtension(config, state);
    const path = await bootstrapServer(config, state);

    return path;
}

async function bootstrapExtension(config: Config, state: PersistentState): Promise<void> {
    if (config.package.releaseTag === null) return;
    if (config.channel === "stable") {
        if (config.package.releaseTag === NIGHTLY_TAG) {
            void vscode.window.showWarningMessage(
                `You are running a nightly version of rust-analyzer extension. ` +
                `To switch to stable, uninstall the extension and re-install it from the marketplace`
            );
        }
        return;
    };

    const now = Date.now();
    if (config.package.releaseTag === NIGHTLY_TAG) {
        // Check if we should poll github api for the new nightly version
        // if we haven't done it during the past hour
        const lastCheck = state.lastCheck;

        const anHour = 60 * 60 * 1000;
        const shouldCheckForNewNightly = state.releaseId === undefined || (now - (lastCheck ?? 0)) > anHour;

        if (!shouldCheckForNewNightly) return;
    }

    const release = await fetchRelease("nightly").catch((e) => {
        log.error(e);
        if (state.releaseId === undefined) { // Show error only for the initial download
            vscode.window.showErrorMessage(`Failed to download rust-analyzer nightly ${e}`);
        }
        return undefined;
    });
    if (release === undefined || release.id === state.releaseId) return;

    const userResponse = await vscode.window.showInformationMessage(
        "New version of rust-analyzer (nightly) is available (requires reload).",
        "Update"
    );
    if (userResponse !== "Update") return;

    const artifact = release.assets.find(artifact => artifact.name === "rust-analyzer.vsix");
    assert(!!artifact, `Bad release: ${JSON.stringify(release)}`);

    const dest = path.join(config.globalStoragePath, "rust-analyzer.vsix");
    await download({
        url: artifact.browser_download_url,
        dest,
        progressTitle: "Downloading rust-analyzer extension",
    });

    await vscode.commands.executeCommand("workbench.extensions.installExtension", vscode.Uri.file(dest));
    await fs.unlink(dest);

    await state.updateReleaseId(release.id);
    await state.updateLastCheck(now);
    await vscode.commands.executeCommand("workbench.action.reloadWindow");
}

async function bootstrapServer(config: Config, state: PersistentState): Promise<string> {
    const path = await getServer(config, state);
    if (!path) {
        throw new Error(
            "Rust Analyzer Language Server is not available. " +
            "Please, ensure its [proper installation](https://rust-analyzer.github.io/manual.html#installation)."
        );
    }

    log.info("Using server binary at", path);

    if (!isValidExecutable(path)) {
        throw new Error(`Failed to execute ${path} --version`);
    }

    return path;
}

async function patchelf(dest: PathLike): Promise<void> {
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: "Patching rust-analyzer for NixOS"
        },
        async (progress, _) => {
            const expression = `
            {src, pkgs ? import <nixpkgs> {}}:
                pkgs.stdenv.mkDerivation {
                    name = "rust-analyzer";
                    inherit src;
                    phases = [ "installPhase" "fixupPhase" ];
                    installPhase = "cp $src $out";
                    fixupPhase = ''
                    chmod 755 $out
                    patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" $out
                    '';
                }
            `;
            const origFile = dest + "-orig";
            await fs.rename(dest, origFile);
            progress.report({ message: "Patching executable", increment: 20 });
            await new Promise((resolve, reject) => {
                const handle = exec(`nix-build -E - --arg src '${origFile}' -o ${dest}`,
                    (err, stdout, stderr) => {
                        if (err != null) {
                            reject(Error(stderr));
                        } else {
                            resolve(stdout);
                        }
                    });
                handle.stdin?.write(expression);
                handle.stdin?.end();
            });
            await fs.unlink(origFile);
        }
    );
}

async function getServer(config: Config, state: PersistentState): Promise<string | undefined> {
    const explicitPath = process.env.__RA_LSP_SERVER_DEBUG ?? config.serverPath;
    if (explicitPath) {
        if (explicitPath.startsWith("~/")) {
            return os.homedir() + explicitPath.slice("~".length);
        }
        return explicitPath;
    };
    if (config.package.releaseTag === null) return "rust-analyzer";

    let platform: string | undefined;
    if (process.arch === "x64" || process.arch === "ia32") {
        if (process.platform === "linux") platform = "linux";
        if (process.platform === "darwin") platform = "mac";
        if (process.platform === "win32") platform = "windows";
    }
    if (platform === undefined) {
        vscode.window.showErrorMessage(
            "Unfortunately we don't ship binaries for your platform yet. " +
            "You need to manually clone rust-analyzer repository and " +
            "run `cargo xtask install --server` to build the language server from sources. " +
            "If you feel that your platform should be supported, please create an issue " +
            "about that [here](https://github.com/rust-analyzer/rust-analyzer/issues) and we " +
            "will consider it."
        );
        return undefined;
    }
    const ext = platform === "windows" ? ".exe" : "";
    const dest = path.join(config.globalStoragePath, `rust-analyzer-${platform}${ext}`);
    const exists = await fs.stat(dest).then(() => true, () => false);
    if (!exists) {
        await state.updateServerVersion(undefined);
    }

    if (state.serverVersion === config.package.version) return dest;

    if (config.askBeforeDownload) {
        const userResponse = await vscode.window.showInformationMessage(
            `Language server version ${config.package.version} for rust-analyzer is not installed.`,
            "Download now"
        );
        if (userResponse !== "Download now") return dest;
    }

    const release = await fetchRelease(config.package.releaseTag);
    const artifact = release.assets.find(artifact => artifact.name === `rust-analyzer-${platform}.gz`);
    assert(!!artifact, `Bad release: ${JSON.stringify(release)}`);

    // Unlinking the exe file before moving new one on its place should prevent ETXTBSY error.
    await fs.unlink(dest).catch(err => {
        if (err.code !== "ENOENT") throw err;
    });

    await download({
        url: artifact.browser_download_url,
        dest,
        progressTitle: "Downloading rust-analyzer server",
        gunzip: true,
        mode: 0o755
    });

    // Patching executable if that's NixOS.
    if (await fs.stat("/etc/nixos").then(_ => true).catch(_ => false)) {
        await patchelf(dest);
    }

    await state.updateServerVersion(config.package.version);
    return dest;
}
