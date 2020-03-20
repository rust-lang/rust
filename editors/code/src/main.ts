import * as vscode from 'vscode';
import * as path from "path";
import * as os from "os";
import { promises as fs } from "fs";

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { activateStatusDisplay } from './status_display';
import { Ctx } from './ctx';
import { activateHighlighting } from './highlighting';
import { Config, NIGHTLY_TAG } from './config';
import { log, assert } from './util';
import { PersistentState } from './persistent_state';
import { fetchRelease, download } from './net';
import { spawnSync } from 'child_process';

let ctx: Ctx | undefined;

export async function activate(context: vscode.ExtensionContext) {
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
    const serverPath = await bootstrap(config, state);

    // Note: we try to start the server before we activate type hints so that it
    // registers its `onDidChangeDocument` handler before us.
    //
    // This a horribly, horribly wrong way to deal with this problem.
    ctx = await Ctx.create(config, context, serverPath);

    // Commands which invokes manually via command palette, shortcut, etc.
    ctx.registerCommand('reload', (ctx) => {
        return async () => {
            vscode.window.showInformationMessage('Reloading rust-analyzer...');
            // @DanTup maneuver
            // https://github.com/microsoft/vscode/issues/45774#issuecomment-373423895
            await deactivate();
            for (const sub of ctx.subscriptions) {
                try {
                    sub.dispose();
                } catch (e) {
                    log.error(e);
                }
            }
            await activate(context);
        };
    });

    ctx.registerCommand('analyzerStatus', commands.analyzerStatus);
    ctx.registerCommand('collectGarbage', commands.collectGarbage);
    ctx.registerCommand('matchingBrace', commands.matchingBrace);
    ctx.registerCommand('joinLines', commands.joinLines);
    ctx.registerCommand('parentModule', commands.parentModule);
    ctx.registerCommand('syntaxTree', commands.syntaxTree);
    ctx.registerCommand('expandMacro', commands.expandMacro);
    ctx.registerCommand('run', commands.run);

    defaultOnEnter.dispose();
    ctx.registerCommand('onEnter', commands.onEnter);

    ctx.registerCommand('ssr', commands.ssr);
    ctx.registerCommand('serverVersion', commands.serverVersion);

    // Internal commands which are invoked by the server.
    ctx.registerCommand('runSingle', commands.runSingle);
    ctx.registerCommand('debugSingle', commands.debugSingle);
    ctx.registerCommand('showReferences', commands.showReferences);
    ctx.registerCommand('applySourceChange', commands.applySourceChange);
    ctx.registerCommand('selectAndApplySourceChange', commands.selectAndApplySourceChange);

    activateStatusDisplay(ctx);

    if (!ctx.config.highlightingSemanticTokens) {
        activateHighlighting(ctx);
    }
    activateInlayHints(ctx);
}

export async function deactivate() {
    await ctx?.client?.stop();
    ctx = undefined;
}

async function bootstrap(config: Config, state: PersistentState): Promise<string> {
    await fs.mkdir(config.globalStoragePath, { recursive: true });

    await bootstrapExtension(config, state);
    const path = await bootstrapServer(config, state);

    return path;
}

async function bootstrapExtension(config: Config, state: PersistentState): Promise<void> {
    if (config.releaseTag === undefined) return;
    if (config.channel === "stable") {
        if (config.releaseTag === NIGHTLY_TAG) {
            vscode.window.showWarningMessage(`You are running a nightly version of rust-analyzer extension.
To switch to stable, uninstall the extension and re-install it from the marketplace`);
        }
        return;
    };

    const lastCheck = state.lastCheck;
    const now = Date.now();

    const anHour = 60 * 60 * 1000;
    const shouldDownloadNightly = state.releaseId === undefined || (now - (lastCheck ?? 0)) > anHour;

    if (!shouldDownloadNightly) return;

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
    await download(artifact.browser_download_url, dest, "Downloading rust-analyzer extension");

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

    const res = spawnSync(path, ["--version"], { encoding: 'utf8' });
    log.debug("Checked binary availability via --version", res);
    log.debug(res, "--version output:", res.output);
    if (res.status !== 0) {
        throw new Error(
            `Failed to execute ${path} --version`
        );
    }

    return path;
}

async function getServer(config: Config, state: PersistentState): Promise<string | undefined> {
    const explicitPath = process.env.__RA_LSP_SERVER_DEBUG ?? config.serverPath;
    if (explicitPath) {
        if (explicitPath.startsWith("~/")) {
            return os.homedir() + explicitPath.slice("~".length);
        }
        return explicitPath;
    };
    if (config.releaseTag === undefined) return "rust-analyzer";

    let binaryName: string | undefined = undefined;
    if (process.arch === "x64" || process.arch === "x32") {
        if (process.platform === "linux") binaryName = "rust-analyzer-linux";
        if (process.platform === "darwin") binaryName = "rust-analyzer-mac";
        if (process.platform === "win32") binaryName = "rust-analyzer-windows.exe";
    }
    if (binaryName === undefined) {
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

    const dest = path.join(config.globalStoragePath, binaryName);
    const exists = await fs.stat(dest).then(() => true, () => false);
    if (!exists) {
        await state.updateServerVersion(undefined);
    }

    if (state.serverVersion === config.packageJsonVersion) return dest;

    if (config.askBeforeDownload) {
        const userResponse = await vscode.window.showInformationMessage(
            `Language server version ${config.packageJsonVersion} for rust-analyzer is not installed.`,
            "Download now"
        );
        if (userResponse !== "Download now") return dest;
    }

    const release = await fetchRelease(config.releaseTag);
    const artifact = release.assets.find(artifact => artifact.name === binaryName);
    assert(!!artifact, `Bad release: ${JSON.stringify(release)}`);

    await download(artifact.browser_download_url, dest, "Downloading rust-analyzer server", { mode: 0o755 });
    await state.updateServerVersion(config.packageJsonVersion);
    return dest;
}
