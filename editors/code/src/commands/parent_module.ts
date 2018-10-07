import * as vscode from 'vscode';

import { Location, TextDocumentIdentifier } from 'vscode-languageclient';
import { Server } from '../server';

export async function handle() {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId != 'rust') { return; }
    const request: TextDocumentIdentifier = {
        uri: editor.document.uri.toString(),
    };
    const response = await Server.client.sendRequest<Location[]>('m/parentModule', request);
    const loc = response[0];
    if (loc == null) { return; }
    const uri = Server.client.protocol2CodeConverter.asUri(loc.uri);
    const range = Server.client.protocol2CodeConverter.asRange(loc.range);

    const doc = await vscode.workspace.openTextDocument(uri);
    const e = await vscode.window.showTextDocument(doc);
    e.selection = new vscode.Selection(range.start, range.start);
    e.revealRange(range, vscode.TextEditorRevealType.InCenter);
}
