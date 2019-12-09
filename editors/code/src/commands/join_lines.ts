import * as vscode from 'vscode';

import { Range, TextDocumentIdentifier } from 'vscode-languageclient';
import { Server } from '../server';
import {
    handle as applySourceChange,
    SourceChange,
} from './apply_source_change';

interface JoinLinesParams {
    textDocument: TextDocumentIdentifier;
    range: Range;
}

export async function handle() {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId !== 'rust') {
        return;
    }
    const request: JoinLinesParams = {
        range: Server.client.code2ProtocolConverter.asRange(editor.selection),
        textDocument: { uri: editor.document.uri.toString() },
    };
    const change = await Server.client.sendRequest<SourceChange>(
        'rust-analyzer/joinLines',
        request,
    );
    await applySourceChange(change);
}
