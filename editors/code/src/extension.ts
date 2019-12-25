import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import * as commands from './commands';
import { ExpandMacroContentProvider } from './commands/expand_macro';
import { HintsUpdater } from './commands/inlay_hints';
import { SyntaxTreeContentProvider } from './commands/syntaxTree';
import { StatusDisplay } from './commands/watch_status';
import * as events from './events';
import * as notifications from './notifications';
import { Server } from './server';

export async function activate(context: vscode.ExtensionContext) {
    function disposeOnDeactivation(disposable: vscode.Disposable) {
        context.subscriptions.push(disposable);
    }

    function registerCommand(name: string, f: any) {
        disposeOnDeactivation(vscode.commands.registerCommand(name, f));
    }
    function overrideCommand(
        name: string,
        f: (...args: any[]) => Promise<boolean>,
    ) {
        const defaultCmd = `default:${name}`;
        const original = (...args: any[]) =>
            vscode.commands.executeCommand(defaultCmd, ...args);

        try {
            registerCommand(name, async (...args: any[]) => {
                const editor = vscode.window.activeTextEditor;
                if (
                    !editor ||
                    !editor.document ||
                    editor.document.languageId !== 'rust'
                ) {
                    return await original(...args);
                }
                if (!(await f(...args))) {
                    return await original(...args);
                }
            });
        } catch (_) {
            vscode.window.showWarningMessage(
                'Enhanced typing feature is disabled because of incompatibility with VIM extension, consider turning off rust-analyzer.enableEnhancedTyping: https://github.com/rust-analyzer/rust-analyzer/blob/master/docs/user/README.md#settings',
            );
        }
    }

    // Commands are requests from vscode to the language server
    registerCommand(
        'rust-analyzer.analyzerStatus',
        commands.analyzerStatus.makeCommand(context),
    );
    registerCommand('rust-analyzer.collectGarbage', () =>
        Server.client.sendRequest<null>('rust-analyzer/collectGarbage', null),
    );
    registerCommand(
        'rust-analyzer.matchingBrace',
        commands.matchingBrace.handle,
    );
    registerCommand('rust-analyzer.joinLines', commands.joinLines.handle);
    registerCommand('rust-analyzer.parentModule', commands.parentModule.handle);
    registerCommand('rust-analyzer.run', commands.runnables.handle);
    // Unlike the above this does not send requests to the language server
    registerCommand('rust-analyzer.runSingle', commands.runnables.handleSingle);
    registerCommand(
        'rust-analyzer.applySourceChange',
        commands.applySourceChange.handle,
    );
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
        overrideCommand('type', commands.onEnter.handle);
    }

    const watchStatus = new StatusDisplay(
        Server.config.cargoWatchOptions.command,
    );
    disposeOnDeactivation(watchStatus);

    // Notifications are events triggered by the language server
    const allNotifications: Iterable<[
        string,
        lc.GenericNotificationHandler,
    ]> = [
        [
            'rust-analyzer/publishDecorations',
            notifications.publishDecorations.handle,
        ],
        [
            '$/progress',
            params => watchStatus.handleProgressNotification(params),
        ],
    ];
    const syntaxTreeContentProvider = new SyntaxTreeContentProvider();
    const expandMacroContentProvider = new ExpandMacroContentProvider();

    // The events below are plain old javascript events, triggered and handled by vscode
    vscode.window.onDidChangeActiveTextEditor(
        events.changeActiveTextEditor.makeHandler(syntaxTreeContentProvider),
    );

    disposeOnDeactivation(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer',
            syntaxTreeContentProvider,
        ),
    );
    disposeOnDeactivation(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer',
            expandMacroContentProvider,
        ),
    );

    registerCommand(
        'rust-analyzer.syntaxTree',
        commands.syntaxTree.createHandle(syntaxTreeContentProvider),
    );
    registerCommand(
        'rust-analyzer.expandMacro',
        commands.expandMacro.createHandle(expandMacroContentProvider),
    );

    vscode.workspace.onDidChangeTextDocument(
        events.changeTextDocument.createHandler(syntaxTreeContentProvider),
        null,
        context.subscriptions,
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
