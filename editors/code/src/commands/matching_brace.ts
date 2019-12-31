import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Ctx, Cmd } from '../ctx';

export function matchingBrace(ctx: Ctx): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        if (!editor || !client) return;

        const request: FindMatchingBraceParams = {
            textDocument: { uri: editor.document.uri.toString() },
            offsets: editor.selections.map(s =>
                client.code2ProtocolConverter.asPosition(s.active),
            ),
        };
        const response = await client.sendRequest<lc.Position[]>(
            'rust-analyzer/findMatchingBrace',
            request,
        );
        editor.selections = editor.selections.map((sel, idx) => {
            const active = client.protocol2CodeConverter.asPosition(
                response[idx],
            );
            const anchor = sel.isEmpty ? active : sel.anchor;
            return new vscode.Selection(anchor, active);
        });
        editor.revealRange(editor.selection);
    };
}

interface FindMatchingBraceParams {
    textDocument: lc.TextDocumentIdentifier;
    offsets: lc.Position[];
}
