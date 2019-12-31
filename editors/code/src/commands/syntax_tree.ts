import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

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
        ctx.subscriptions,
    );

    vscode.window.onDidChangeActiveTextEditor(
        (editor: vscode.TextEditor | undefined) => {
            if (!editor || editor.document.languageId !== 'rust') return;
            tdcp.eventEmitter.fire(tdcp.uri);
        },
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
function afterLs(f: () => any) {
    setTimeout(f, 10);
}

interface SyntaxTreeParams {
    textDocument: lc.TextDocumentIdentifier;
    range?: lc.Range;
}

class TextDocumentContentProvider
    implements vscode.TextDocumentContentProvider {
    private ctx: Ctx;
    uri = vscode.Uri.parse('rust-analyzer://syntaxtree');
    eventEmitter = new vscode.EventEmitter<vscode.Uri>();

    constructor(ctx: Ctx) {
        this.ctx = ctx;
    }

    provideTextDocumentContent(uri: vscode.Uri): vscode.ProviderResult<string> {
        const editor = vscode.window.activeTextEditor;
        const client = this.ctx.client;
        if (!editor || !client) return '';

        let range: lc.Range | undefined;

        // When the range based query is enabled we take the range of the selection
        if (uri.query === 'range=true') {
            range = editor.selection.isEmpty
                ? undefined
                : client.code2ProtocolConverter.asRange(editor.selection);
        }

        const request: SyntaxTreeParams = {
            textDocument: { uri: editor.document.uri.toString() },
            range,
        };
        return client.sendRequest<string>(
            'rust-analyzer/syntaxTree',
            request,
        );
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}
