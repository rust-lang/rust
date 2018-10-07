import * as vscode from 'vscode';

import { TextDocumentIdentifier, Position } from "vscode-languageclient";
import { Server } from '../server';

interface FindMatchingBraceParams {
    textDocument: TextDocumentIdentifier;
    offsets: Position[];
}

export async function handle() {
    let editor = vscode.window.activeTextEditor
    if (editor == null || editor.document.languageId != "rust") return
    let request: FindMatchingBraceParams = {
        textDocument: { uri: editor.document.uri.toString() },
        offsets: editor.selections.map((s) => {
            return Server.client.code2ProtocolConverter.asPosition(s.active)
        })
    }
    let response = await Server.client.sendRequest<Position[]>("m/findMatchingBrace", request)
    editor.selections = editor.selections.map((sel, idx) => {
        let active = Server.client.protocol2CodeConverter.asPosition(response[idx])
        let anchor = sel.isEmpty ? active : sel.anchor
        return new vscode.Selection(anchor, active)
    })
    editor.revealRange(editor.selection)
};
