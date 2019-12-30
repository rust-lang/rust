import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import * as commands from './commands';
import { HintsUpdater } from './commands/inlay_hints';
import { StatusDisplay } from './commands/watch_status';
import * as events from './events';
import * as notifications from './notifications';
import { Server } from './server';
import { Ctx } from './ctx';

let ctx!: Ctx;

export async function activate(context: vscode.ExtensionContext) {
    ctx = new Ctx(context);
    ctx.registerCommand('analyzerStatus', commands.analyzerStatus);
    ctx.registerCommand('collectGarbage', commands.collectGarbage);
    ctx.registerCommand('matchingBrace', commands.matchingBrace);
    ctx.registerCommand('joinLines', commands.joinLines);
    ctx.registerCommand('parentModule', commands.parentModule);
    ctx.registerCommand('syntaxTree', commands.syntaxTree);
    ctx.registerCommand('expandMacro', commands.expandMacro);

    function disposeOnDeactivation(disposable: vscode.Disposable) {
        context.subscriptions.push(disposable);
    }

    function registerCommand(name: string, f: any) {
        disposeOnDeactivation(vscode.commands.registerCommand(name, f));
    }

    // Commands are requests from vscode to the language server
    registerCommand('rust-analyzer.run', commands.runnables.handle);
    // Unlike the above this does not send requests to the language server
    registerCommand('rust-analyzer.runSingle', commands.runnables.handleSingle);
    registerCommand(
        'rust-analyzer.showReferences',
        (uri: string, position: lc.Position, locations: lc.Location[]) => {
            vscode.commands.executeCommand(
                'editor.action.showReferences',
                vscode.Uri.parse(uri),
                Server.client.protocol2CodeConverter.asPosition(position),
                locations.map(Server.client.protocol2CodeConverter.asLocation),
            );
        },
    );

    if (Server.config.enableEnhancedTyping) {
        ctx.overrideCommand('type', commands.onEnter);
    }

    const watchStatus = new StatusDisplay(
        Server.config.cargoWatchOptions.command,
    );
    disposeOnDeactivation(watchStatus);

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
        const hintsUpdater = new HintsUpdater();
        hintsUpdater.refreshHintsForVisibleEditors().then(() => {
            // vscode may ignore top level hintsUpdater.refreshHintsForVisibleEditors()
            // so update the hints once when the focus changes to guarantee their presence
            let editorChangeDisposable: vscode.Disposable | null = null;
            editorChangeDisposable = vscode.window.onDidChangeActiveTextEditor(
                _ => {
                    if (editorChangeDisposable !== null) {
                        editorChangeDisposable.dispose();
                    }
                    return hintsUpdater.refreshHintsForVisibleEditors();
                },
            );

            disposeOnDeactivation(
                vscode.window.onDidChangeVisibleTextEditors(_ =>
                    hintsUpdater.refreshHintsForVisibleEditors(),
                ),
            );
            disposeOnDeactivation(
                vscode.workspace.onDidChangeTextDocument(e =>
                    hintsUpdater.refreshHintsForVisibleEditors(e),
                ),
            );
            disposeOnDeactivation(
                vscode.workspace.onDidChangeConfiguration(_ =>
                    hintsUpdater.toggleHintsDisplay(
                        Server.config.displayInlayHints,
                    ),
                ),
            );
        });
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
