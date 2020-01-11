import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Ctx } from './ctx';

export interface SourceChange {
    label: string;
    workspaceEdit: lc.WorkspaceEdit;
    cursorPosition?: lc.TextDocumentPositionParams;
}

export async function applySourceChange(ctx: Ctx, change: SourceChange) {
    const client = ctx.client;
    if (!client) return;

    const wsEdit = client.protocol2CodeConverter.asWorkspaceEdit(
        change.workspaceEdit,
    );
    let created;
    let moved;
    if (change.workspaceEdit.documentChanges) {
        for (const docChange of change.workspaceEdit.documentChanges) {
            if (lc.CreateFile.is(docChange)) {
                created = docChange.uri;
            } else if (lc.RenameFile.is(docChange)) {
                moved = docChange.newUri;
            }
        }
    }
    const toOpen = created || moved;
    const toReveal = change.cursorPosition;
    await vscode.workspace.applyEdit(wsEdit);
    if (toOpen) {
        const toOpenUri = vscode.Uri.parse(toOpen);
        const doc = await vscode.workspace.openTextDocument(toOpenUri);
        await vscode.window.showTextDocument(doc);
    } else if (toReveal) {
        const uri = client.protocol2CodeConverter.asUri(
            toReveal.textDocument.uri,
        );
        const position = client.protocol2CodeConverter.asPosition(
            toReveal.position,
        );
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.uri.toString() !== uri.toString()) {
            return;
        }
        if (!editor.selection.isEmpty) {
            return;
        }
        editor.selection = new vscode.Selection(position, position);
        editor.revealRange(
            new vscode.Range(position, position),
            vscode.TextEditorRevealType.Default,
        );
    }
}
