import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import * as ra from '../rust-analyzer-api';

import { Ctx, Cmd } from '../ctx';
import * as sourceChange from '../source_change';
import { assert } from '../util';

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

export function selectAndApplySourceChange(ctx: Ctx): Cmd {
    return async (changes: ra.SourceChange[]) => {
        if (changes.length === 1) {
            await sourceChange.applySourceChange(ctx, changes[0]);
        } else if (changes.length > 0) {
            const selectedChange = await vscode.window.showQuickPick(changes);
            if (!selectedChange) return;
            await sourceChange.applySourceChange(ctx, selectedChange);
        }
    };
}

export function applySnippetWorkspaceEdit(_ctx: Ctx): Cmd {
    return async (edit: vscode.WorkspaceEdit) => {
        assert(edit.entries().length === 1, `bad ws edit: ${JSON.stringify(edit)}`);
        const [uri, edits] = edit.entries()[0];

        const editor = vscode.window.visibleTextEditors.find((it) => it.document.uri.toString() === uri.toString());
        if (!editor) return;

        let editWithSnippet: vscode.TextEdit | undefined = undefined;
        let lineDelta = 0;
        await editor.edit((builder) => {
            for (const indel of edits) {
                if (indel.newText.indexOf('$0') !== -1) {
                    editWithSnippet = indel;
                } else {
                    if (!editWithSnippet) {
                        lineDelta = (indel.newText.match(/\n/g) || []).length - (indel.range.end.line - indel.range.start.line);
                    }
                    builder.replace(indel.range, indel.newText);
                }
            }
        });
        if (editWithSnippet) {
            const snip = editWithSnippet as vscode.TextEdit;
            const range = snip.range.with(
                snip.range.start.with(snip.range.start.line + lineDelta),
                snip.range.end.with(snip.range.end.line + lineDelta),
            );
            await editor.insertSnippet(new vscode.SnippetString(snip.newText), range);
        }
    };
}
