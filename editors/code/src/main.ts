import * as vscode from 'vscode';

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { activateStatusDisplay } from './status_display';
import { Server } from './server';
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

    // Internal commands which are invoked by the server.
    ctx.registerCommand('runSingle', commands.runSingle);
    ctx.registerCommand('showReferences', commands.showReferences);
    ctx.registerCommand('applySourceChange', commands.applySourceChange);

    if (ctx.config.enableEnhancedTyping) {
        ctx.overrideCommand('type', commands.onEnter);
    }

    const startServer = () => Server.start();
    const reloadCommand = () => reloadServer(startServer);

    vscode.commands.registerCommand('rust-analyzer.reload', reloadCommand);

    // Start the language server, finally!
    try {
        await startServer();
    } catch (e) {
        vscode.window.showErrorMessage(e.message);
    }

    activateStatusDisplay(ctx);
    activateHighlighting(ctx);

    if (ctx.config.displayInlayHints) {
        activateInlayHints(ctx);
    }
}

export function deactivate(): Thenable<void> {
    if (!Server.client) {
        return Promise.resolve();
    }
    return Server.client.stop();
}

async function reloadServer(startServer: () => Promise<void>) {
    if (Server.client != null) {
        vscode.window.showInformationMessage('Reloading rust-analyzer...');
        await Server.client.stop();
        await startServer();
    }
}
