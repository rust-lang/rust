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

export async function applySnippetWorkspaceEdit(edit: vscode.WorkspaceEdit) {
    assert(edit.entries().length === 1, `bad ws edit: ${JSON.stringify(edit)}`);
    const [uri, edits] = edit.entries()[0];

    const editor = vscode.window.visibleTextEditors.find((it) => it.document.uri.toString() === uri.toString());
    if (!editor) return;

    let selection: vscode.Selection | undefined = undefined;
    let lineDelta = 0;
    await editor.edit((builder) => {
        for (const indel of edits) {
            const parsed = parseSnippet(indel.newText);
            if (parsed) {
                const [newText, [placeholderStart, placeholderLength]] = parsed;
                const prefix = newText.substr(0, placeholderStart);
                const lastNewline = prefix.lastIndexOf('\n');

                const startLine = indel.range.start.line + lineDelta + countLines(prefix);
                const startColumn = lastNewline === -1 ?
                    indel.range.start.character + placeholderStart
                    : prefix.length - lastNewline - 1;
                const endColumn = startColumn + placeholderLength;
                selection = new vscode.Selection(
                    new vscode.Position(startLine, startColumn),
                    new vscode.Position(startLine, endColumn),
                );
                builder.replace(indel.range, newText);
            } else {
                lineDelta = countLines(indel.newText) - (indel.range.end.line - indel.range.start.line);
                builder.replace(indel.range, indel.newText);
            }
        }
    });
    if (selection) editor.selection = selection;
}

function parseSnippet(snip: string): [string, [number, number]] | undefined {
    const m = snip.match(/\$(0|\{0:([^}]*)\})/);
    if (!m) return undefined;
    const placeholder = m[2] ?? "";
    const range: [number, number] = [m.index!!, placeholder.length];
    const insert = snip.replace(m[0], placeholder);
    return [insert, range];
}

function countLines(text: string): number {
    return (text.match(/\n/g) || []).length;
}
