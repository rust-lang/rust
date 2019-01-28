import * as vscode from 'vscode';

import { Range, TextDocumentIdentifier } from 'vscode-languageclient';
import { Server } from '../server';

interface ExtendSelectionParams {
    textDocument: TextDocumentIdentifier;
    selections: Range[];
}

interface ExtendSelectionResult {
    selections: Range[];
}

export async function handle() {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId !== 'rust') {
        return;
    }
    const request: ExtendSelectionParams = {
        selections: editor.selections.map(s =>
            Server.client.code2ProtocolConverter.asRange(s)
        ),
        textDocument: { uri: editor.document.uri.toString() }
    };
    const response = await Server.client.sendRequest<ExtendSelectionResult>(
        'rust-analyzer/extendSelection',
        request
    );
    editor.selections = response.selections.map((range: Range) => {
        const r = Server.client.protocol2CodeConverter.asRange(range);
        return new vscode.Selection(r.start, r.end);
    });
}
