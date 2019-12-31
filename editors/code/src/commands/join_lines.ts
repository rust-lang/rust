import * as lc from 'vscode-languageclient';

import { Ctx, Cmd } from '../ctx';
import { applySourceChange, SourceChange } from '../source_change';

export function joinLines(ctx: Ctx): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        if (!editor || !client) return;

        const request: JoinLinesParams = {
            range: client.code2ProtocolConverter.asRange(editor.selection),
            textDocument: { uri: editor.document.uri.toString() },
        };
        const change = await client.sendRequest<SourceChange>(
            'rust-analyzer/joinLines',
            request,
        );
        await applySourceChange(ctx, change);
    };
}

interface JoinLinesParams {
    textDocument: lc.TextDocumentIdentifier;
    range: lc.Range;
}
