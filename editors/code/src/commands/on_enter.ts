import * as lc from 'vscode-languageclient';

import { applySourceChange, SourceChange } from '../source_change';
import { Cmd, Ctx } from '../ctx';

export function onEnter(ctx: Ctx): Cmd {
    return async (event: { text: string }) => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        if (!editor || event.text !== '\n') return false;
        if (!client) return false;

        const request: lc.TextDocumentPositionParams = {
            textDocument: { uri: editor.document.uri.toString() },
            position: client.code2ProtocolConverter.asPosition(
                editor.selection.active,
            ),
        };
        const change = await client.sendRequest<undefined | SourceChange>(
            'rust-analyzer/onEnter',
            request,
        );
        if (!change) return false;

        await applySourceChange(ctx, change);
        return true;
    };
}
