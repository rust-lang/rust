import * as ra from '../rust-analyzer-api';
import * as lc from 'vscode-languageclient';

import { Ctx, Cmd } from '../ctx';

export function joinLines(ctx: Ctx): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        if (!editor || !client) return;

        const items: lc.TextEdit[] = await client.sendRequest(ra.joinLines, {
            ranges: editor.selections.map((it) => client.code2ProtocolConverter.asRange(it)),
            textDocument: { uri: editor.document.uri.toString() },
        });
        editor.edit((builder) => {
            client.protocol2CodeConverter.asTextEdits(items).forEach((edit) => {
                builder.replace(edit.range, edit.newText);
            });
        });
    };
}
