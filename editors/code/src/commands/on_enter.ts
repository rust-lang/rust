import * as vscode from 'vscode';
import * as ra from '../rust-analyzer-api';

import { applySourceChange } from '../source_change';
import { Cmd, Ctx } from '../ctx';

async function handleKeypress(ctx: Ctx) {
    const editor = ctx.activeRustEditor;
    const client = ctx.client;

    if (!editor || !client) return false;

    const change = await client.sendRequest(ra.onEnter, {
        textDocument: { uri: editor.document.uri.toString() },
        position: client.code2ProtocolConverter.asPosition(
            editor.selection.active,
        ),
    }).catch(_error => {
        // client.logFailedRequest(OnEnterRequest.type, error);
        return null;
    });
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
