import * as vscode from "vscode";

import { assert } from "./util";
import { unwrapUndefinable } from "./undefinable";

export async function applySnippetWorkspaceEdit(edit: vscode.WorkspaceEdit) {
    if (edit.entries().length === 1) {
        const [uri, edits] = unwrapUndefinable(edit.entries()[0]);
        const editor = await editorFromUri(uri);
        if (editor) await applySnippetTextEdits(editor, edits);
        return;
    }
    for (const [uri, edits] of edit.entries()) {
        const editor = await editorFromUri(uri);
        if (editor) {
            await editor.edit((builder) => {
                for (const indel of edits) {
                    assert(
                        !parseSnippet(indel.newText),
                        `bad ws edit: snippet received with multiple edits: ${JSON.stringify(
                            edit,
                        )}`,
                    );
                    builder.replace(indel.range, indel.newText);
                }
            });
        }
    }
}

async function editorFromUri(uri: vscode.Uri): Promise<vscode.TextEditor | undefined> {
    if (vscode.window.activeTextEditor?.document.uri !== uri) {
        // `vscode.window.visibleTextEditors` only contains editors whose contents are being displayed
        await vscode.window.showTextDocument(uri, {});
    }
    return vscode.window.visibleTextEditors.find(
        (it) => it.document.uri.toString() === uri.toString(),
    );
}

export async function applySnippetTextEdits(editor: vscode.TextEditor, edits: vscode.TextEdit[]) {
    const selections: vscode.Selection[] = [];
    let lineDelta = 0;
    await editor.edit((builder) => {
        for (const indel of edits) {
            const parsed = parseSnippet(indel.newText);
            if (parsed) {
                const [newText, [placeholderStart, placeholderLength]] = parsed;
                const prefix = newText.substr(0, placeholderStart);
                const lastNewline = prefix.lastIndexOf("\n");

                const startLine = indel.range.start.line + lineDelta + countLines(prefix);
                const startColumn =
                    lastNewline === -1
                        ? indel.range.start.character + placeholderStart
                        : prefix.length - lastNewline - 1;
                const endColumn = startColumn + placeholderLength;
                selections.push(
                    new vscode.Selection(
                        new vscode.Position(startLine, startColumn),
                        new vscode.Position(startLine, endColumn),
                    ),
                );
                builder.replace(indel.range, newText);
            } else {
                builder.replace(indel.range, indel.newText);
            }
            lineDelta +=
                countLines(indel.newText) - (indel.range.end.line - indel.range.start.line);
        }
    });
    if (selections.length > 0) editor.selections = selections;
    if (selections.length === 1) {
        const selection = unwrapUndefinable(selections[0]);
        editor.revealRange(selection, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
    }
}

function parseSnippet(snip: string): [string, [number, number]] | undefined {
    const m = snip.match(/\$(0|\{0:([^}]*)\})/);
    if (!m) return undefined;
    const placeholder = m[2] ?? "";
    if (m.index == null) return undefined;
    const range: [number, number] = [m.index, placeholder.length];
    const insert = snip.replace(m[0], placeholder);
    return [insert, range];
}

function countLines(text: string): number {
    return (text.match(/\n/g) || []).length;
}
