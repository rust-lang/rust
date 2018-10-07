import * as vscode from 'vscode';

import * as commands from './commands';
import { TextDocumentContentProvider } from './commands/syntaxTree';
import * as events from './events';
import { Server } from './server';

export function activate(context: vscode.ExtensionContext) {
    function disposeOnDeactivation(disposable: vscode.Disposable) {
        context.subscriptions.push(disposable);
    }

    function registerCommand(name: string, f: any) {
        disposeOnDeactivation(vscode.commands.registerCommand(name, f));
    }

    registerCommand('ra-lsp.syntaxTree', commands.syntaxTree.handle);
    registerCommand('ra-lsp.extendSelection', commands.extendSelection.handle);
    registerCommand('ra-lsp.matchingBrace', commands.matchingBrace.handle);
    registerCommand('ra-lsp.joinLines', commands.joinLines.handle);
    registerCommand('ra-lsp.parentModule', commands.parentModule.handle);
    registerCommand('ra-lsp.run', commands.runnables.handle);
    registerCommand('ra-lsp.applySourceChange', commands.applySourceChange.handle);

    const textDocumentContentProvider = new TextDocumentContentProvider();
    disposeOnDeactivation(vscode.workspace.registerTextDocumentContentProvider(
        'ra-lsp',
        textDocumentContentProvider,
    ));

    Server.start();

    vscode.workspace.onDidChangeTextDocument(
        events.changeTextDocument.createHandler(textDocumentContentProvider),
        null,
        context.subscriptions);
    vscode.window.onDidChangeActiveTextEditor(events.changeActiveTextEditor.handle);
}

export function deactivate(): Thenable<void> {
    if (!Server.client) {
        return Promise.resolve();
    }
    return Server.client.stop();
}
