import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import * as commands from './commands';
import { activateInlayHints } from './inlay_hints';
import { StatusDisplay } from './status_display';
import * as events from './events';
import * as notifications from './notifications';
import { Server } from './server';
import { Ctx } from './ctx';

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

    if (Server.config.enableEnhancedTyping) {
        ctx.overrideCommand('type', commands.onEnter);
    }

    const watchStatus = new StatusDisplay(
        Server.config.cargoWatchOptions.command,
    );
    ctx.pushCleanup(watchStatus);

    // Notifications are events triggered by the language server
    const allNotifications: [string, lc.GenericNotificationHandler][] = [
        [
            'rust-analyzer/publishDecorations',
            notifications.publishDecorations.handle,
        ],
        [
            '$/progress',
            params => watchStatus.handleProgressNotification(params),
        ],
    ];

    // The events below are plain old javascript events, triggered and handled by vscode
    vscode.window.onDidChangeActiveTextEditor(
        events.changeActiveTextEditor.makeHandler(),
    );

    const startServer = () => Server.start(allNotifications);
    const reloadCommand = () => reloadServer(startServer);

    vscode.commands.registerCommand('rust-analyzer.reload', reloadCommand);

    // Start the language server, finally!
    try {
        await startServer();
    } catch (e) {
        vscode.window.showErrorMessage(e.message);
    }

    if (Server.config.displayInlayHints) {
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
