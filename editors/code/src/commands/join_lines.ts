import * as vscode from 'vscode';

import { TextDocumentIdentifier, Range } from "vscode-languageclient";
import { Server } from '../server';
import { handle as applySourceChange, SourceChange } from './apply_source_change';

interface JoinLinesParams {
    textDocument: TextDocumentIdentifier;
    range: Range;
}

export async function handle() {
    let editor = vscode.window.activeTextEditor
    if (editor == null || editor.document.languageId != "rust") return
    let request: JoinLinesParams = {
        textDocument: { uri: editor.document.uri.toString() },
        range: Server.client.code2ProtocolConverter.asRange(editor.selection),
    }
    let change = await Server.client.sendRequest<SourceChange>("m/joinLines", request)
    await applySourceChange(change)
}
