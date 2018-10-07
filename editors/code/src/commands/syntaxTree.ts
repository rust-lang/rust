import * as vscode from 'vscode';
import { TextDocumentIdentifier } from 'vscode-languageclient';

import { Server } from '../server';

export const syntaxTreeUri = vscode.Uri.parse('ra-lsp://syntaxtree');

export class TextDocumentContentProvider implements vscode.TextDocumentContentProvider {
    public eventEmitter = new vscode.EventEmitter<vscode.Uri>()
    public syntaxTree: string = "Not available"

    public provideTextDocumentContent(uri: vscode.Uri): vscode.ProviderResult<string> {
        let editor = vscode.window.activeTextEditor;
        if (editor == null) return ""
        let request: SyntaxTreeParams = {
            textDocument: { uri: editor.document.uri.toString() }
        };
        return Server.client.sendRequest<SyntaxTreeResult>("m/syntaxTree", request);
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event
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
    let document = await vscode.workspace.openTextDocument(syntaxTreeUri)
    return vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true)
}
