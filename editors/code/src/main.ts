import * as vscode from 'vscode';
import * as path from "path";
import * as os from "os";
import { promises as fs, PathLike } from "fs";

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { Ctx } from './ctx';
import { Config } from './config';
import { log, assert, isValidExecutable, isRustDocument } from './util';
import { PersistentState } from './persistent_state';
import { fetchRelease, download } from './net';
import { activateTaskProvider } from './tasks';
import { setContextValue } from './util';
import { exec, spawnSync } from 'child_process';

let ctx: Ctx | undefined;

const RUST_PROJECT_CONTEXT_NAME = "inRustProject";

export async function activate(context: vscode.ExtensionContext) {
    // VS Code doesn't show a notification when an extension fails to activate
    // so we do it ourselves.
    await tryActivate(context).catch(err => {
        void vscode.window.showErrorMessage(`Cannot activate rust-analyzer: ${err.message}`);
        throw err;
    });
}

async function tryActivate(context: vscode.ExtensionContext) {
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

    if ((vscode.workspace.workspaceFolders || []).length === 0) {
        const rustDocuments = vscode.workspace.textDocuments.filter(document => isRustDocument(document));
        if (rustDocuments.length > 0) {
            ctx = await Ctx.create(config, context, serverPath, { kind: 'Detached Files', files: rustDocuments });
        } else {
            throw new Error("no rust files are opened");
        }
    } else {
        // Note: we try to start the server before we activate type hints so that it
        // registers its `onDidChangeDocument` handler before us.
        //
        // This a horribly, horribly wrong way to deal with this problem.
        ctx = await Ctx.create(config, context, serverPath, { kind: "Workspace Folder" });
        ctx.pushCleanup(activateTaskProvider(ctx.config));
    }
    await initCommonContext(context, ctx);

    activateInlayHints(ctx);
    warnAboutExtensionConflicts();

    vscode.workspace.onDidChangeConfiguration(
        _ => ctx?.client?.sendNotification('workspace/didChangeConfiguration', { settings: "" }),
        null,
        ctx.subscriptions,
    );
}

async function initCommonContext(context: vscode.ExtensionContext, ctx: Ctx) {
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

    await setContextValue(RUST_PROJECT_CONTEXT_NAME, true);

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

    ctx.registerCommand('updateGithubToken', ctx => async () => {
        await queryForGithubToken(new PersistentState(ctx.globalState));
    });

    ctx.registerCommand('analyzerStatus', commands.analyzerStatus);
    ctx.registerCommand('memoryUsage', commands.memoryUsage);
    ctx.registerCommand('reloadWorkspace', commands.reloadWorkspace);
    ctx.registerCommand('matchingBrace', commands.matchingBrace);
    ctx.registerCommand('joinLines', commands.joinLines);
    ctx.registerCommand('parentModule', commands.parentModule);
    ctx.registerCommand('syntaxTree', commands.syntaxTree);
    ctx.registerCommand('viewHir', commands.viewHir);
    ctx.registerCommand('viewItemTree', commands.viewItemTree);
    ctx.registerCommand('viewCrateGraph', commands.viewCrateGraph);
    ctx.registerCommand('expandMacro', commands.expandMacro);
    ctx.registerCommand('run', commands.run);
    ctx.registerCommand('copyRunCommandLine', commands.copyRunCommandLine);
    ctx.registerCommand('debug', commands.debug);
    ctx.registerCommand('newDebugConfig', commands.newDebugConfig);
    ctx.registerCommand('openDocs', commands.openDocs);
    ctx.registerCommand('openCargoToml', commands.openCargoToml);
    ctx.registerCommand('peekTests', commands.peekTests);
    ctx.registerCommand('moveItemUp', commands.moveItemUp);
    ctx.registerCommand('moveItemDown', commands.moveItemDown);

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
}

export async function deactivate() {
    await setContextValue(RUST_PROJECT_CONTEXT_NAME, undefined);
    await ctx?.client.stop();
    ctx = undefined;
}

async function bootstrap(config: Config, state: PersistentState): Promise<string> {
    await fs.mkdir(config.globalStoragePath, { recursive: true });

    if (!config.currentExtensionIsNightly) {
        await state.updateNightlyReleaseId(undefined);
    }
    await bootstrapExtension(config, state);
    const path = await bootstrapServer(config, state);
    return path;
}

async function bootstrapExtension(config: Config, state: PersistentState): Promise<void> {
    if (config.package.releaseTag === null) return;
    if (config.channel === "stable") {
        if (config.currentExtensionIsNightly) {
            void vscode.window.showWarningMessage(
                `You are running a nightly version of rust-analyzer extension. ` +
                `To switch to stable, uninstall the extension and re-install it from the marketplace`
            );
        }
        return;
    };
    if (serverPath(config)) return;

    const now = Date.now();
    const isInitialNightlyDownload = state.nightlyReleaseId === undefined;
    if (config.currentExtensionIsNightly) {
        // Check if we should poll github api for the new nightly version
        // if we haven't done it during the past hour
        const lastCheck = state.lastCheck;

        const anHour = 60 * 60 * 1000;
        const shouldCheckForNewNightly = isInitialNightlyDownload || (now - (lastCheck ?? 0)) > anHour;

        if (!shouldCheckForNewNightly) return;
    }

    const latestNightlyRelease = await downloadWithRetryDialog(state, async () => {
        return await fetchRelease("nightly", state.githubToken, config.httpProxy);
    }).catch(async (e) => {
        log.error(e);
        if (isInitialNightlyDownload) {
            await vscode.window.showErrorMessage(`Failed to download rust-analyzer nightly: ${e}`);
        }
        return;
    });
    if (latestNightlyRelease === undefined) {
        if (isInitialNightlyDownload) {
            await vscode.window.showErrorMessage("Failed to download rust-analyzer nightly: empty release contents returned");
        }
        return;
    }
    if (config.currentExtensionIsNightly && latestNightlyRelease.id === state.nightlyReleaseId) return;

    const userResponse = await vscode.window.showInformationMessage(
        "New version of rust-analyzer (nightly) is available (requires reload).",
        "Update"
    );
    if (userResponse !== "Update") return;

    const artifact = latestNightlyRelease.assets.find(artifact => artifact.name === "rust-analyzer.vsix");
    assert(!!artifact, `Bad release: ${JSON.stringify(latestNightlyRelease)}`);

    const dest = path.join(config.globalStoragePath, "rust-analyzer.vsix");

    await downloadWithRetryDialog(state, async () => {
        await download({
            url: artifact.browser_download_url,
            dest,
            progressTitle: "Downloading rust-analyzer extension",
            httpProxy: config.httpProxy,
        });
    });

    await vscode.commands.executeCommand("workbench.extensions.installExtension", vscode.Uri.file(dest));
    await fs.unlink(dest);

    await state.updateNightlyReleaseId(latestNightlyRelease.id);
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
            const origFile = dest + "-orig";
            await fs.rename(dest, origFile);
            progress.report({ message: "Patching executable", increment: 20 });
            await new Promise((resolve, reject) => {
                const handle = exec(`nix-build -E - --argstr srcStr '${origFile}' -o '${dest}'`,
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
    const explicitPath = serverPath(config);
    if (explicitPath) {
        if (explicitPath.startsWith("~/")) {
            return os.homedir() + explicitPath.slice("~".length);
        }
        return explicitPath;
    };
    if (config.package.releaseTag === null) return "rust-analyzer";

    const platforms: { [key: string]: string } = {
        "ia32 win32": "x86_64-pc-windows-msvc",
        "x64 win32": "x86_64-pc-windows-msvc",
        "x64 linux": "x86_64-unknown-linux-gnu",
        "x64 darwin": "x86_64-apple-darwin",
        "arm64 win32": "aarch64-pc-windows-msvc",
        "arm64 linux": "aarch64-unknown-linux-gnu",
        "arm64 darwin": "aarch64-apple-darwin",
    };
    let platform = platforms[`${process.arch} ${process.platform}`];
    if (platform === undefined) {
        await vscode.window.showErrorMessage(
            "Unfortunately we don't ship binaries for your platform yet. " +
            "You need to manually clone rust-analyzer repository and " +
            "run `cargo xtask install --server` to build the language server from sources. " +
            "If you feel that your platform should be supported, please create an issue " +
            "about that [here](https://github.com/rust-analyzer/rust-analyzer/issues) and we " +
            "will consider it."
        );
        return undefined;
    }
    if (platform === "x86_64-unknown-linux-gnu" && isMusl()) {
        platform = "x86_64-unknown-linux-musl";
    }
    const ext = platform.indexOf("-windows-") !== -1 ? ".exe" : "";
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

    const releaseTag = config.package.releaseTag;
    const release = await downloadWithRetryDialog(state, async () => {
        return await fetchRelease(releaseTag, state.githubToken, config.httpProxy);
    });
    const artifact = release.assets.find(artifact => artifact.name === `rust-analyzer-${platform}.gz`);
    assert(!!artifact, `Bad release: ${JSON.stringify(release)}`);

    await downloadWithRetryDialog(state, async () => {
        await download({
            url: artifact.browser_download_url,
            dest,
            progressTitle: "Downloading rust-analyzer server",
            gunzip: true,
            mode: 0o755,
            httpProxy: config.httpProxy,
        });
    });

    // Patching executable if that's NixOS.
    if (await isNixOs()) {
        await patchelf(dest);
    }

    await state.updateServerVersion(config.package.version);
    return dest;
}

function serverPath(config: Config): string | null {
    return process.env.__RA_LSP_SERVER_DEBUG ?? config.serverPath;
}

async function isNixOs(): Promise<boolean> {
    try {
        const contents = await fs.readFile("/etc/os-release");
        return contents.indexOf("ID=nixos") !== -1;
    } catch (e) {
        return false;
    }
}

function isMusl(): boolean {
    // We can detect Alpine by checking `/etc/os-release` but not Void Linux musl.
    // Instead, we run `ldd` since it advertises the libc which it belongs to.
    const res = spawnSync("ldd", ["--version"]);
    return res.stderr != null && res.stderr.indexOf("musl libc") >= 0;
}

async function downloadWithRetryDialog<T>(state: PersistentState, downloadFunc: () => Promise<T>): Promise<T> {
    while (true) {
        try {
            return await downloadFunc();
        } catch (e) {
            const selected = await vscode.window.showErrorMessage("Failed to download: " + e.message, {}, {
                title: "Update Github Auth Token",
                updateToken: true,
            }, {
                title: "Retry download",
                retry: true,
            }, {
                title: "Dismiss",
            });

            if (selected?.updateToken) {
                await queryForGithubToken(state);
                continue;
            } else if (selected?.retry) {
                continue;
            }
            throw e;
        };
    }
}

async function queryForGithubToken(state: PersistentState): Promise<void> {
    const githubTokenOptions: vscode.InputBoxOptions = {
        value: state.githubToken,
        password: true,
        prompt: `
            This dialog allows to store a Github authorization token.
            The usage of an authorization token will increase the rate
            limit on the use of Github APIs and can thereby prevent getting
            throttled.
            Auth tokens can be created at https://github.com/settings/tokens`,
    };

    const newToken = await vscode.window.showInputBox(githubTokenOptions);
    if (newToken === undefined) {
        // The user aborted the dialog => Do not update the stored token
        return;
    }

    if (newToken === "") {
        log.info("Clearing github token");
        await state.updateGithubToken(undefined);
    } else {
        log.info("Storing new github token");
        await state.updateGithubToken(newToken);
    }
}

function warnAboutExtensionConflicts() {
    const conflicting = [
        ["rust-analyzer", "matklad.rust-analyzer"],
        ["Rust", "rust-lang.rust"],
        ["Rust", "kalitaalexey.vscode-rust"],
    ];

    const found = conflicting.filter(
        nameId => vscode.extensions.getExtension(nameId[1]) !== undefined);

    if (found.length > 1) {
        const fst = found[0];
        const sec = found[1];
        vscode.window.showWarningMessage(
            `You have both the ${fst[0]} (${fst[1]}) and ${sec[0]} (${sec[1]}) ` +
            "plugins enabled. These are known to conflict and cause various functions of " +
            "both plugins to not work correctly. You should disable one of them.", "Got it")
            .then(() => { }, console.error);
    };
}
