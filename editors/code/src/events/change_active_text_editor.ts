import { TextEditor } from 'vscode';
import { TextDocumentIdentifier } from 'vscode-languageclient';
import { Decoration } from '../highlighting';
import { Server } from '../server';

export function makeHandler() {
    return async function handle(editor: TextEditor | undefined) {
        if (!editor || editor.document.languageId !== 'rust') {
            return;
        }

        if (!Server.config.highlightingOn) {
            return;
        }

        const params: TextDocumentIdentifier = {
            uri: editor.document.uri.toString(),
        };
        const decorations = await Server.client.sendRequest<Decoration[]>(
            'rust-analyzer/decorationsRequest',
            params,
        );
        Server.highlighter.setHighlights(editor, decorations);
    };
}
