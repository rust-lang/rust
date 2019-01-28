import * as vscode from 'vscode';

import * as lc from 'vscode-languageclient';
import { Server } from '../server';

export async function handle() {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId !== 'rust') {
        return;
    }
    const request: lc.TextDocumentPositionParams = {
        textDocument: { uri: editor.document.uri.toString() },
        position: Server.client.code2ProtocolConverter.asPosition(
            editor.selection.active
        )
    };
    const response = await Server.client.sendRequest<lc.Location[]>(
        'rust-analyzer/parentModule',
        request
    );
    const loc = response[0];
    if (loc == null) {
        return;
    }
    const uri = Server.client.protocol2CodeConverter.asUri(loc.uri);
    const range = Server.client.protocol2CodeConverter.asRange(loc.range);

    const doc = await vscode.workspace.openTextDocument(uri);
    const e = await vscode.window.showTextDocument(doc);
    e.selection = new vscode.Selection(range.start, range.start);
    e.revealRange(range, vscode.TextEditorRevealType.InCenter);
}
