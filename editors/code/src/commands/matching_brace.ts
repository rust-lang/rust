import * as vscode from 'vscode';
import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd } from '../ctx';

export function matchingBrace(ctx: Ctx): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        if (!editor || !client) return;

        const response = await client.sendRequest(ra.findMatchingBrace, {
            textDocument: { uri: editor.document.uri.toString() },
            offsets: editor.selections.map(s =>
                client.code2ProtocolConverter.asPosition(s.active),
            ),
        });
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
