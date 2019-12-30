import * as lc from 'vscode-languageclient';
import { applySourceChange, SourceChange } from '../source_change';
import { Cmd, Ctx } from '../ctx';

export function onEnter(ctx: Ctx): Cmd {
    return async (event: { text: string }) => {
        const editor = ctx.activeRustEditor;
        if (!editor || event.text !== '\n') return false;

        const request: lc.TextDocumentPositionParams = {
            textDocument: { uri: editor.document.uri.toString() },
            position: ctx.client.code2ProtocolConverter.asPosition(
                editor.selection.active,
            ),
        };
        const change = await ctx.client.sendRequest<undefined | SourceChange>(
            'rust-analyzer/onEnter',
            request,
        );
        if (!change) return false;

        await applySourceChange(ctx, change);
        return true;
    };
}
