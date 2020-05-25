import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd } from '../ctx';
import { applySnippetWorkspaceEdit } from '../snippets';

export * from './analyzer_status';
export * from './matching_brace';
export * from './join_lines';
export * from './on_enter';
export * from './parent_module';
export * from './syntax_tree';
export * from './expand_macro';
export * from './runnables';
export * from './ssr';
export * from './server_version';
export * from './toggle_inlay_hints';

export function collectGarbage(ctx: Ctx): Cmd {
    return async () => ctx.client.sendRequest(ra.collectGarbage, null);
}

export function showReferences(ctx: Ctx): Cmd {
    return (uri: string, position: lc.Position, locations: lc.Location[]) => {
        const client = ctx.client;
        if (client) {
            vscode.commands.executeCommand(
                'editor.action.showReferences',
                vscode.Uri.parse(uri),
                client.protocol2CodeConverter.asPosition(position),
                locations.map(client.protocol2CodeConverter.asLocation),
            );
        }
    };
}

export function applySourceChange(ctx: Ctx): Cmd {
    return async (change: ra.SourceChange) => {
        await sourceChange.applySourceChange(ctx, change);
    };
}

export function applyActionGroup(_ctx: Ctx): Cmd {
    return async (actions: { label: string; edit: vscode.WorkspaceEdit }[]) => {
        const selectedAction = await vscode.window.showQuickPick(actions);
        if (!selectedAction) return;
        await applySnippetWorkspaceEdit(selectedAction.edit);
    };
}

export function applySnippetWorkspaceEditCommand(_ctx: Ctx): Cmd {
    return async (edit: vscode.WorkspaceEdit) => {
        await applySnippetWorkspaceEdit(edit);
    };
}
