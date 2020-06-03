import * as vscode from 'vscode';

import { assert } from './util';

export async function applySnippetWorkspaceEdit(edit: vscode.WorkspaceEdit) {
    assert(edit.entries().length === 1, `bad ws edit: ${JSON.stringify(edit)}`);
    const [uri, edits] = edit.entries()[0];

    const editor = vscode.window.visibleTextEditors.find((it) => it.document.uri.toString() === uri.toString());
    if (!editor) return;
    await applySnippetTextEdits(editor, edits);
}

export async function applySnippetTextEdits(editor: vscode.TextEditor, edits: vscode.TextEdit[]) {
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
