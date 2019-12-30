import * as vscode from 'vscode';
import { Position, TextDocumentIdentifier } from 'vscode-languageclient';
import { Ctx, Cmd } from '../ctx';

export function matchingBrace(ctx: Ctx): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        if (!editor) {
            return;
        }
        const request: FindMatchingBraceParams = {
            textDocument: { uri: editor.document.uri.toString() },
            offsets: editor.selections.map(s => ctx.client.code2ProtocolConverter.asPosition(s.active)),
        };
        const response = await ctx.client.sendRequest<Position[]>(
            'rust-analyzer/findMatchingBrace',
            request,
        );
        editor.selections = editor.selections.map((sel, idx) => {
            const active = ctx.client.protocol2CodeConverter.asPosition(
                response[idx],
            );
            const anchor = sel.isEmpty ? active : sel.anchor;
            return new vscode.Selection(anchor, active);
        });
        editor.revealRange(editor.selection);
    }
}

interface FindMatchingBraceParams {
    textDocument: TextDocumentIdentifier;
    offsets: Position[];
}
