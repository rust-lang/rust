import * as vscode from 'vscode';
import { TextDocumentIdentifier } from 'vscode-languageclient';

import { Server } from '../server';

export const syntaxTreeUri = vscode.Uri.parse('rust-analyzer://syntaxtree');

export class TextDocumentContentProvider
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
        const request: SyntaxTreeParams = {
            textDocument: { uri: editor.document.uri.toString() }
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
}

type SyntaxTreeResult = string;

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export async function handle() {
    const document = await vscode.workspace.openTextDocument(syntaxTreeUri);
    return vscode.window.showTextDocument(
        document,
        vscode.ViewColumn.Two,
        true
    );
}
