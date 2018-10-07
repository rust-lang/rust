import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient'

import { Server } from '../server';

interface FileSystemEdit {
    type: string;
    uri?: string;
    src?: string;
    dst?: string;
}

export interface SourceChange {
    label: string,
    sourceFileEdits: lc.TextDocumentEdit[],
    fileSystemEdits: FileSystemEdit[],
    cursorPosition?: lc.TextDocumentPositionParams,
}

export async function handle(change: SourceChange) {
    console.log(`applySOurceChange ${JSON.stringify(change)}`)
    let wsEdit = new vscode.WorkspaceEdit()
    for (let sourceEdit of change.sourceFileEdits) {
        let uri = Server.client.protocol2CodeConverter.asUri(sourceEdit.textDocument.uri)
        let edits = Server.client.protocol2CodeConverter.asTextEdits(sourceEdit.edits)
        wsEdit.set(uri, edits)
    }
    let created;
    let moved;
    for (let fsEdit of change.fileSystemEdits) {
        if (fsEdit.type == "createFile") {
            let uri = vscode.Uri.parse(fsEdit.uri!)
            wsEdit.createFile(uri)
            created = uri
        } else if (fsEdit.type == "moveFile") {
            let src = vscode.Uri.parse(fsEdit.src!)
            let dst = vscode.Uri.parse(fsEdit.dst!)
            wsEdit.renameFile(src, dst)
            moved = dst
        } else {
            console.error(`unknown op: ${JSON.stringify(fsEdit)}`)
        }
    }
    let toOpen = created || moved
    let toReveal = change.cursorPosition
    await vscode.workspace.applyEdit(wsEdit)
    if (toOpen) {
        let doc = await vscode.workspace.openTextDocument(toOpen)
        await vscode.window.showTextDocument(doc)
    } else if (toReveal) {
        let uri = Server.client.protocol2CodeConverter.asUri(toReveal.textDocument.uri)
        let position = Server.client.protocol2CodeConverter.asPosition(toReveal.position)
        let editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.uri.toString() != uri.toString()) return
        if (!editor.selection.isEmpty) return
        editor!.selection = new vscode.Selection(position, position)
    }
}
