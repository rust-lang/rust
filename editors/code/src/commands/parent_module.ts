import * as vscode from 'vscode';

import { TextDocumentIdentifier, Location } from "vscode-languageclient";
import { Server } from '../server';

export async function handle() {
    let editor = vscode.window.activeTextEditor
    if (editor == null || editor.document.languageId != "rust") return
    let request: TextDocumentIdentifier = {
        uri: editor.document.uri.toString()
    }
    let response = await Server.client.sendRequest<Location[]>("m/parentModule", request)
    let loc = response[0]
    if (loc == null) return
    let uri = Server.client.protocol2CodeConverter.asUri(loc.uri)
    let range = Server.client.protocol2CodeConverter.asRange(loc.range)

    let doc = await vscode.workspace.openTextDocument(uri)
    let e = await vscode.window.showTextDocument(doc)
    e.selection = new vscode.Selection(range.start, range.start)
    e.revealRange(range, vscode.TextEditorRevealType.InCenter)
}
