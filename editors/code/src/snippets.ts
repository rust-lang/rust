import * as vscode from "vscode";

import { assert } from "./util";
import { unwrapUndefinable } from "./undefinable";

export type SnippetTextDocumentEdit = [vscode.Uri, (vscode.TextEdit | vscode.SnippetTextEdit)[]];

export async function applySnippetWorkspaceEdit(
    edit: vscode.WorkspaceEdit,
    editEntries: SnippetTextDocumentEdit[],
) {
    if (editEntries.length === 1) {
        const [uri, edits] = unwrapUndefinable(editEntries[0]);
        const editor = await editorFromUri(uri);
        if (editor) {
            edit.set(uri, edits);
            await vscode.workspace.applyEdit(edit);
        }
        return;
    }
    for (const [uri, edits] of editEntries) {
        const editor = await editorFromUri(uri);
        if (editor) {
            await editor.edit((builder) => {
                for (const indel of edits) {
                    assert(
                        !(indel instanceof vscode.SnippetTextEdit),
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
    const edit = new vscode.WorkspaceEdit();
    edit.set(editor.document.uri, toSnippetTextEdits(edits));
    await vscode.workspace.applyEdit(edit);
}

function hasSnippet(snip: string): boolean {
    const m = snip.match(/\$\d+|\{\d+:[^}]*\}/);
    return m != null;
}

function toSnippetTextEdits(
    edits: vscode.TextEdit[],
): (vscode.TextEdit | vscode.SnippetTextEdit)[] {
    return edits.map((textEdit) => {
        // Note: text edits without any snippets are returned as-is instead of
        // being wrapped in a SnippetTextEdit, as otherwise it would be
        // treated as if it had a tab stop at the end.
        if (hasSnippet(textEdit.newText)) {
            return new vscode.SnippetTextEdit(
                textEdit.range,
                new vscode.SnippetString(textEdit.newText),
            );
        } else {
            return textEdit;
        }
    });
}
