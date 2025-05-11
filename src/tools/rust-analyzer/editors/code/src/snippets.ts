import * as vscode from "vscode";

import { assert, unwrapUndefinable } from "./util";

export type SnippetTextDocumentEdit = [vscode.Uri, (vscode.TextEdit | vscode.SnippetTextEdit)[]];

export async function applySnippetWorkspaceEdit(
    edit: vscode.WorkspaceEdit,
    editEntries: SnippetTextDocumentEdit[],
) {
    if (editEntries.length === 1) {
        const [uri, edits] = unwrapUndefinable(editEntries[0]);
        const editor = await editorFromUri(uri);
        if (editor) {
            edit.set(uri, removeLeadingWhitespace(editor, edits));
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
                        `bad ws edit: snippet received with multiple edits: ${JSON.stringify(edit)}`,
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
    const snippetEdits = toSnippetTextEdits(edits);
    edit.set(editor.document.uri, removeLeadingWhitespace(editor, snippetEdits));
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

/**
 * Removes the leading whitespace from snippet edits, so as to not double up
 * on indentation.
 *
 * Snippet edits by default adjust any multi-line snippets to match the
 * indentation of the line to insert at. Unfortunately, we (the server) also
 * include the required indentation to match what we line insert at, so we end
 * up doubling up the indentation. Since there isn't any way to tell vscode to
 * not fixup indentation for us, we instead opt to remove the indentation and
 * then let vscode add it back in.
 *
 * This assumes that the source snippet text edits have the required
 * indentation, but that's okay as even without this workaround and the problem
 * to workaround, those snippet edits would already be inserting at the wrong
 * indentation.
 */
function removeLeadingWhitespace(
    editor: vscode.TextEditor,
    edits: (vscode.TextEdit | vscode.SnippetTextEdit)[],
) {
    return edits.map((edit) => {
        if (edit instanceof vscode.SnippetTextEdit) {
            const snippetEdit: vscode.SnippetTextEdit = edit;
            const firstLineEnd = snippetEdit.snippet.value.indexOf("\n");

            if (firstLineEnd !== -1) {
                // Is a multi-line snippet, remove the indentation which
                // would be added back in by vscode.
                const startLine = editor.document.lineAt(snippetEdit.range.start.line);
                const leadingWhitespace = getLeadingWhitespace(
                    startLine.text,
                    0,
                    startLine.firstNonWhitespaceCharacterIndex,
                );

                const [firstLine, rest] = splitAt(snippetEdit.snippet.value, firstLineEnd + 1);
                const unindentedLines = rest
                    .split("\n")
                    .map((line) => line.replace(leadingWhitespace, ""))
                    .join("\n");

                snippetEdit.snippet.value = firstLine + unindentedLines;
            }

            return snippetEdit;
        } else {
            return edit;
        }
    });
}

// based on https://github.com/microsoft/vscode/blob/main/src/vs/base/common/strings.ts#L284
function getLeadingWhitespace(str: string, start: number = 0, end: number = str.length): string {
    for (let i = start; i < end; i++) {
        const chCode = str.charCodeAt(i);
        if (chCode !== " ".charCodeAt(0) && chCode !== " ".charCodeAt(0)) {
            return str.substring(start, i);
        }
    }
    return str.substring(start, end);
}

function splitAt(str: string, index: number): [string, string] {
    return [str.substring(0, index), str.substring(index)];
}
