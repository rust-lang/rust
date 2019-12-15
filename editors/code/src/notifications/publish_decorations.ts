import * as vscode from 'vscode';

import { Decoration } from '../highlighting';
import { Server } from '../server';

export interface PublishDecorationsParams {
    uri: string;
    decorations: Decoration[];
}

export function handle(params: PublishDecorationsParams) {
    const targetEditor = vscode.window.visibleTextEditors.find(
        editor => {
            const unescapedUri = unescape(editor.document.uri.toString());
            // Unescaped URI should be something like:
            // file:///c:/Workspace/ra-test/src/main.rs
            // RA server might send it with the drive letter uppercased, so we force only the drive letter to lowercase.
            const uriWithLowercasedDrive = params.uri.substr(0, 8) + params.uri[8].toLowerCase() + params.uri.substr(9);
            return unescapedUri === uriWithLowercasedDrive
        }
    );

    if (!Server.config.highlightingOn || !targetEditor) {
        return;
    }

    Server.highlighter.setHighlights(targetEditor, params.decorations);
}
