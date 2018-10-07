import * as vscode from 'vscode';

import { TextDocumentIdentifier, Range } from "vscode-languageclient";
import { Server } from '../server';

interface ExtendSelectionParams {
    textDocument: TextDocumentIdentifier;
    selections: Range[];
}

interface ExtendSelectionResult {
    selections: Range[];
}

export async function handle() {
    let editor = vscode.window.activeTextEditor
    if (editor == null || editor.document.languageId != "rust") return
    let request: ExtendSelectionParams = {
        textDocument: { uri: editor.document.uri.toString() },
        selections: editor.selections.map((s) => {
            return Server.client.code2ProtocolConverter.asRange(s)
        })
    }
    let response = await Server.client.sendRequest<ExtendSelectionResult>("m/extendSelection", request)
    editor.selections = response.selections.map((range: Range) => {
        let r = Server.client.protocol2CodeConverter.asRange(range)
        return new vscode.Selection(r.start, r.end)
    })
}
