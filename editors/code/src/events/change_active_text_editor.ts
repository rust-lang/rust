import { TextEditor } from 'vscode';
import { TextDocumentIdentifier } from 'vscode-languageclient';

import {
    SyntaxTreeContentProvider,
    syntaxTreeUri
} from '../commands/syntaxTree';
import { Decoration } from '../highlighting';
import { Server } from '../server';

export function makeHandler(syntaxTreeProvider: SyntaxTreeContentProvider) {
    return async function handle(editor: TextEditor | undefined) {
        if (!editor || editor.document.languageId !== 'rust') {
            return;
        }

        syntaxTreeProvider.eventEmitter.fire(syntaxTreeUri);

        if (!Server.config.highlightingOn) {
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
    };
}
