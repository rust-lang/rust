import { TextEditor } from 'vscode';
import { TextDocumentIdentifier } from 'vscode-languageclient';

import { Decoration } from '../highlighting';
import { Server } from '../server';

export async function handle(editor: TextEditor | undefined) {
    if (
        !Server.config.highlightingOn ||
        !editor ||
        editor.document.languageId !== 'rust'
    ) {
        return;
    }
    const params: TextDocumentIdentifier = {
        uri: editor.document.uri.toString()
    };
    const decorations = await Server.client.sendRequest<Decoration[]>(
        'rust-analyzer/decorationsRequest',
        params
    );
    Server.highlighter.setHighlights(editor, decorations);
}
