import * as vscode from 'vscode';

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { activateStatusDisplay } from './status_display';
import { Ctx } from './ctx';
import { activateHighlighting } from './highlighting';

let ctx!: Ctx;

export async function activate(context: vscode.ExtensionContext) {
    ctx = new Ctx(context);

    // Commands which invokes manually via command pallet, shortcut, etc.
    ctx.registerCommand('analyzerStatus', commands.analyzerStatus);
    ctx.registerCommand('collectGarbage', commands.collectGarbage);
    ctx.registerCommand('matchingBrace', commands.matchingBrace);
    ctx.registerCommand('joinLines', commands.joinLines);
    ctx.registerCommand('parentModule', commands.parentModule);
    ctx.registerCommand('syntaxTree', commands.syntaxTree);
    ctx.registerCommand('expandMacro', commands.expandMacro);
    ctx.registerCommand('run', commands.run);
    ctx.registerCommand('reload', commands.reload);

    // Internal commands which are invoked by the server.
    ctx.registerCommand('runSingle', commands.runSingle);
    ctx.registerCommand('showReferences', commands.showReferences);
    ctx.registerCommand('applySourceChange', commands.applySourceChange);

    if (ctx.config.enableEnhancedTyping) {
        ctx.overrideCommand('type', commands.onEnter);
    }
    activateStatusDisplay(ctx);

    activateHighlighting(ctx);
    // Note: we try to start the server before we activate type hints so that it
    // registers its `onDidChangeDocument` handler before us.
    //
    // This a horribly, horribly wrong way to deal with this problem.
    try {
        await ctx.restartServer();
    } catch (e) {
        vscode.window.showErrorMessage(e.message);
    }
    activateInlayHints(ctx);
}

export async function deactivate() {
    await ctx?.client?.stop();
}
