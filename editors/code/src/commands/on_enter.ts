import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import { Server } from '../server';
import { handle as applySourceChange, SourceChange } from './apply_source_change';

interface OnEnterParams {
    textDocument: lc.TextDocumentIdentifier;
    position: lc.Position;
}

export async function handle(event: { text: string }): Promise<boolean> {
    const editor = vscode.window.activeTextEditor;
    if (editor == null || editor.document.languageId !== 'rust' || event.text !== '\n') {
        return false;
    }
    const request: OnEnterParams = {
        textDocument: { uri: editor.document.uri.toString() },
        position: Server.client.code2ProtocolConverter.asPosition(editor.selection.active),
    };
    const change = await Server.client.sendRequest<undefined | SourceChange>(
        'm/onEnter',
        request
    );
    if (!change) {
        return false;
    }
    await applySourceChange(change);
    return true
}
