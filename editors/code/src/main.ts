import * as vscode from 'vscode';

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { activateStatusDisplay } from './status_display';
import { Ctx } from './ctx';
import { activateHighlighting } from './highlighting';
import { ensureServerBinary } from './installation/server';
import { Config } from './config';
import { log } from './util';
import { ensureProperExtensionVersion } from './installation/extension';

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

    vscode.workspace.onDidChangeConfiguration(() => ensureProperExtensionVersion(config).catch(log.error));

    // Don't await the user response here, otherwise we will block the lsp server bootstrap
    void ensureProperExtensionVersion(config).catch(log.error);

    const serverPath = await ensureServerBinary(config);

    if (serverPath == null) {
        throw new Error(
            "Rust Analyzer Language Server is not available. " +
            "Please, ensure its [proper installation](https://rust-analyzer.github.io/manual.html#installation)."
        );
    }

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
