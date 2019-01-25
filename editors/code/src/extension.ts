import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import * as commands from './commands';
import { TextDocumentContentProvider } from './commands/syntaxTree';
import * as events from './events';
import * as notifications from './notifications';
import { Server } from './server';

export function activate(context: vscode.ExtensionContext) {
    function disposeOnDeactivation(disposable: vscode.Disposable) {
        context.subscriptions.push(disposable);
    }

    function registerCommand(name: string, f: any) {
        disposeOnDeactivation(vscode.commands.registerCommand(name, f));
    }
    function overrideCommand(
        name: string,
        f: (...args: any[]) => Promise<boolean>
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
                'Enhanced typing feature is disabled because of incompatibility with VIM extension'
            );
        }
    }

    // Commands are requests from vscode to the language server
    registerCommand(
        'ra-lsp.analyzerStatus',
        commands.analyzerStatus.makeCommand(context)
    );
    registerCommand('ra-lsp.collectGarbage', () =>
        Server.client.sendRequest<null>('ra/collectGarbage', null)
    );
    registerCommand('ra-lsp.syntaxTree', commands.syntaxTree.handle);
    registerCommand('ra-lsp.extendSelection', commands.extendSelection.handle);
    registerCommand('ra-lsp.matchingBrace', commands.matchingBrace.handle);
    registerCommand('ra-lsp.joinLines', commands.joinLines.handle);
    registerCommand('ra-lsp.parentModule', commands.parentModule.handle);
    registerCommand('ra-lsp.run', commands.runnables.handle);
    registerCommand(
        'ra-lsp.applySourceChange',
        commands.applySourceChange.handle
    );
    overrideCommand('type', commands.onEnter.handle);

    // Unlike the above this does not send requests to the language server
    registerCommand('ra-lsp.run-single', commands.runnables.handleSingle);

    // Notifications are events triggered by the language server
    const allNotifications: Iterable<
        [string, lc.GenericNotificationHandler]
    > = [['m/publishDecorations', notifications.publishDecorations.handle]];

    // The events below are plain old javascript events, triggered and handled by vscode
    vscode.window.onDidChangeActiveTextEditor(
        events.changeActiveTextEditor.handle
    );

    const textDocumentContentProvider = new TextDocumentContentProvider();
    disposeOnDeactivation(
        vscode.workspace.registerTextDocumentContentProvider(
            'ra-lsp',
            textDocumentContentProvider
        )
    );

    vscode.workspace.onDidChangeTextDocument(
        events.changeTextDocument.createHandler(textDocumentContentProvider),
        null,
        context.subscriptions
    );

    // Start the language server, finally!
    Server.start(allNotifications);
}

export function deactivate(): Thenable<void> {
    if (!Server.client) {
        return Promise.resolve();
    }
    return Server.client.stop();
}
