import * as vscode from 'vscode';
import { Range, TextDocumentIdentifier } from 'vscode-languageclient';

import { Server } from '../server';

export const syntaxTreeUri = vscode.Uri.parse('rust-analyzer://syntaxtree');

export class SyntaxTreeContentProvider
    implements vscode.TextDocumentContentProvider {
    public eventEmitter = new vscode.EventEmitter<vscode.Uri>();
    public syntaxTree: string = 'Not available';

    public provideTextDocumentContent(
        uri: vscode.Uri
    ): vscode.ProviderResult<string> {
        const editor = vscode.window.activeTextEditor;
        if (editor == null) {
            return '';
        }

        let range: Range | undefined;

        // When the range based query is enabled we take the range of the selection
        if (uri.query === 'range=true') {
            range = editor.selection.isEmpty
                ? undefined
                : Server.client.code2ProtocolConverter.asRange(
                      editor.selection
                  );
        }

        const request: SyntaxTreeParams = {
            textDocument: { uri: editor.document.uri.toString() },
            range
        };
        return Server.client.sendRequest<SyntaxTreeResult>(
            'rust-analyzer/syntaxTree',
            request
        );
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event;
    }
}

interface SyntaxTreeParams {
    textDocument: TextDocumentIdentifier;
    range?: Range;
}

type SyntaxTreeResult = string;

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export function createHandle(provider: SyntaxTreeContentProvider) {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        const rangeEnabled = !!(editor && !editor.selection.isEmpty);

        const uri = rangeEnabled
            ? vscode.Uri.parse(`${syntaxTreeUri.toString()}?range=true`)
            : syntaxTreeUri;

        const document = await vscode.workspace.openTextDocument(uri);

        provider.eventEmitter.fire(uri);

        return vscode.window.showTextDocument(
            document,
            vscode.ViewColumn.Two,
            true
        );
    };
}
