import { Range, TextDocumentIdentifier } from 'vscode-languageclient';
import { Ctx, Cmd } from '../ctx';
import { applySourceChange, SourceChange } from '../source_change';

export function joinLines(ctx: Ctx): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        const request: JoinLinesParams = {
            range: ctx.client.code2ProtocolConverter.asRange(editor.selection),
            textDocument: { uri: editor.document.uri.toString() },
        };
        const change = await ctx.client.sendRequest<SourceChange>(
            'rust-analyzer/joinLines',
            request,
        );
        await applySourceChange(ctx, change);
    };
}

interface JoinLinesParams {
    textDocument: TextDocumentIdentifier;
    range: Range;
}
