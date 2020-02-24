import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { applySourceChange, SourceChange } from '../source_change';
import { Cmd, Ctx } from '../ctx';

async function handleKeypress(ctx: Ctx) {
    const editor = ctx.activeRustEditor;
    const client = ctx.client;

    if (!editor || !client) return false;

    const request: lc.TextDocumentPositionParams = {
        textDocument: { uri: editor.document.uri.toString() },
        position: client.code2ProtocolConverter.asPosition(
            editor.selection.active,
        ),
    };
    const change = await client.sendRequest<undefined | SourceChange>(
        'rust-analyzer/onEnter',
        request,
    ).catch(
        (_error: any) => {
            // FIXME: switch to the more modern (?) typed request infrastructure
            // client.logFailedRequest(OnEnterRequest.type, error);
            return Promise.resolve(null);
        }
    );
    if (!change) return false;

    await applySourceChange(ctx, change);
    return true;
}

export function onEnter(ctx: Ctx): Cmd {
    return async () => {
        if (await handleKeypress(ctx)) return;

        await vscode.commands.executeCommand('default:type', { text: '\n' });
    };
}
