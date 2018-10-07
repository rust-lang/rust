import { TextEditor } from "vscode";
import { TextDocumentIdentifier } from "vscode-languageclient";

import { Server } from "../server";
import { Decoration } from "../highlighting";

export async function handle(editor: TextEditor | undefined) {
    if (!Server.config.highlightingOn || !editor || editor.document.languageId != 'rust') return
    let params: TextDocumentIdentifier = {
        uri: editor.document.uri.toString()
    }
    let decorations = await Server.client.sendRequest<Decoration[]>("m/decorationsRequest", params)
    Server.highlighter.setHighlights(editor, decorations)
}