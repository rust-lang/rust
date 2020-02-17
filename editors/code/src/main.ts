import * as vscode from 'vscode';

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { activateStatusDisplay } from './status_display';
import { Ctx } from './ctx';
import { activateHighlighting } from './highlighting';
import { ensureServerBinary } from './installation/server';

let ctx: Ctx | undefined;

export async function activate(context: vscode.ExtensionContext) {
    ctx = new Ctx(context);

    const serverPath = await ensureServerBinary(ctx.config.serverSource);
    if (serverPath == null) {
        throw new Error(
            "Rust Analyzer Language Server is not available. " +
            "Please, ensure its [proper installation](https://github.com/rust-analyzer/rust-analyzer/tree/master/docs/user#vs-code)."
        );
    }

    // Note: we try to start the server before we activate type hints so that it
    // registers its `onDidChangeDocument` handler before us.
    //
    // This a horribly, horribly wrong way to deal with this problem.
    try {
        await ctx.startServer(serverPath);
    } catch (e) {
        vscode.window.showErrorMessage(e.message);
    }

    // Commands which invokes manually via command palette, shortcut, etc.
    ctx.registerCommand('reload', (ctx) => {
        return async () => {
            vscode.window.showInformationMessage('Reloading rust-analyzer...');
            // @DanTup maneuver
            // https://github.com/microsoft/vscode/issues/45774#issuecomment-373423895
            await deactivate()
            for (const sub of ctx.subscriptions) {
                try {
                    sub.dispose();
                } catch (e) {
                    console.error(e);
                }
            }
            await activate(context)
        }
    })

    ctx.registerCommand('analyzerStatus', commands.analyzerStatus);
    ctx.registerCommand('collectGarbage', commands.collectGarbage);
    ctx.registerCommand('matchingBrace', commands.matchingBrace);
    ctx.registerCommand('joinLines', commands.joinLines);
    ctx.registerCommand('parentModule', commands.parentModule);
    ctx.registerCommand('syntaxTree', commands.syntaxTree);
    ctx.registerCommand('expandMacro', commands.expandMacro);
    ctx.registerCommand('run', commands.run);
    ctx.registerCommand('onEnter', commands.onEnter);
    ctx.registerCommand('ssr', commands.ssr)

    // Internal commands which are invoked by the server.
    ctx.registerCommand('runSingle', commands.runSingle);
    ctx.registerCommand('showReferences', commands.showReferences);
    ctx.registerCommand('applySourceChange', commands.applySourceChange);
    ctx.registerCommand('selectAndApplySourceChange', commands.selectAndApplySourceChange);

    activateStatusDisplay(ctx);

    activateHighlighting(ctx);
    activateInlayHints(ctx);
}

export async function deactivate() {
    await ctx?.client?.stop();
    ctx = undefined;
}
