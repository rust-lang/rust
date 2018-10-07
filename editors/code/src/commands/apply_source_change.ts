import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Server } from '../server';

interface FileSystemEdit {
    type: string;
    uri?: string;
    src?: string;
    dst?: string;
}

export interface SourceChange {
    label: string;
    sourceFileEdits: lc.TextDocumentEdit[];
    fileSystemEdits: FileSystemEdit[];
    cursorPosition?: lc.TextDocumentPositionParams;
}

export async function handle(change: SourceChange) {
    console.log(`applySOurceChange ${JSON.stringify(change)}`);
    const wsEdit = new vscode.WorkspaceEdit();
    for (const sourceEdit of change.sourceFileEdits) {
        const uri = Server.client.protocol2CodeConverter.asUri(sourceEdit.textDocument.uri);
        const edits = Server.client.protocol2CodeConverter.asTextEdits(sourceEdit.edits);
        wsEdit.set(uri, edits);
    }
    let created;
    let moved;
    for (const fsEdit of change.fileSystemEdits) {
        if (fsEdit.type == 'createFile') {
            const uri = vscode.Uri.parse(fsEdit.uri!);
            wsEdit.createFile(uri);
            created = uri;
        } else if (fsEdit.type == 'moveFile') {
            const src = vscode.Uri.parse(fsEdit.src!);
            const dst = vscode.Uri.parse(fsEdit.dst!);
            wsEdit.renameFile(src, dst);
            moved = dst;
        } else {
            console.error(`unknown op: ${JSON.stringify(fsEdit)}`);
        }
    }
    const toOpen = created || moved;
    const toReveal = change.cursorPosition;
    await vscode.workspace.applyEdit(wsEdit);
    if (toOpen) {
        const doc = await vscode.workspace.openTextDocument(toOpen);
        await vscode.window.showTextDocument(doc);
    } else if (toReveal) {
        const uri = Server.client.protocol2CodeConverter.asUri(toReveal.textDocument.uri);
        const position = Server.client.protocol2CodeConverter.asPosition(toReveal.position);
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.uri.toString() != uri.toString()) { return; }
        if (!editor.selection.isEmpty) { return; }
        editor!.selection = new vscode.Selection(position, position);
    }
}
