import * as vscode from 'vscode';

import {
    SyntaxTreeContentProvider,
    syntaxTreeUri
} from '../commands/syntaxTree';

export function createHandler(syntaxTreeProvider: SyntaxTreeContentProvider) {
    return (event: vscode.TextDocumentChangeEvent) => {
        const doc = event.document;
        if (doc.languageId !== 'rust') {
            return;
        }
        afterLs(() => {
            syntaxTreeProvider.eventEmitter.fire(syntaxTreeUri);
        });
    };
}

// We need to order this after LS updates, but there's no API for that.
// Hence, good old setTimeout.
function afterLs(f: () => any) {
    setTimeout(f, 10);
}
