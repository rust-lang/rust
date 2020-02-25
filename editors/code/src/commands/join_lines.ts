import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd } from '../ctx';
import { applySourceChange } from '../source_change';

export function joinLines(ctx: Ctx): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        if (!editor || !client) return;

        const change = await client.sendRequest(ra.joinLines, {
            range: client.code2ProtocolConverter.asRange(editor.selection),
            textDocument: { uri: editor.document.uri.toString() },
        });
        await applySourceChange(ctx, change);
    };
}
