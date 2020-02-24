import * as vscode from 'vscode';
import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd } from '../ctx';

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export function syntaxTree(ctx: Ctx): Cmd {
    const tdcp = new TextDocumentContentProvider(ctx);

    ctx.pushCleanup(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer',
            tdcp,
        ),
    );

    vscode.workspace.onDidChangeTextDocument(
        (event: vscode.TextDocumentChangeEvent) => {
            const doc = event.document;
            if (doc.languageId !== 'rust') return;
            afterLs(() => tdcp.eventEmitter.fire(tdcp.uri));
        },
        null,
        ctx.subscriptions,
    );

    vscode.window.onDidChangeActiveTextEditor(
        (editor: vscode.TextEditor | undefined) => {
            if (!editor || editor.document.languageId !== 'rust') return;
            tdcp.eventEmitter.fire(tdcp.uri);
        },
        null,
        ctx.subscriptions,
    );

    return async () => {
        const editor = vscode.window.activeTextEditor;
        const rangeEnabled = !!(editor && !editor.selection.isEmpty);

        const uri = rangeEnabled
            ? vscode.Uri.parse(`${tdcp.uri.toString()}?range=true`)
            : tdcp.uri;

        const document = await vscode.workspace.openTextDocument(uri);

        tdcp.eventEmitter.fire(uri);

        return vscode.window.showTextDocument(
            document,
            vscode.ViewColumn.Two,
            true,
        );
    };
}

// We need to order this after LS updates, but there's no API for that.
// Hence, good old setTimeout.
function afterLs(f: () => void) {
    setTimeout(f, 10);
}


class TextDocumentContentProvider implements vscode.TextDocumentContentProvider {
    uri = vscode.Uri.parse('rust-analyzer://syntaxtree');
    eventEmitter = new vscode.EventEmitter<vscode.Uri>();

    constructor(private readonly ctx: Ctx) {
    }

    provideTextDocumentContent(uri: vscode.Uri): vscode.ProviderResult<string> {
        const editor = vscode.window.activeTextEditor;
        const client = this.ctx.client;
        if (!editor || !client) return '';

        // When the range based query is enabled we take the range of the selection
        const range = uri.query === 'range=true' && !editor.selection.isEmpty
            ? client.code2ProtocolConverter.asRange(editor.selection)
            : null;

        return client.sendRequest(ra.syntaxTree, {
            textDocument: { uri: editor.document.uri.toString() },
            range,
        });
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}
