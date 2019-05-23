import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Server } from './server';

export interface Decoration {
    range: lc.Range;
    tag: string;
}

export class Highlighter {
    private static initDecorations(): Map<
        string,
        vscode.TextEditorDecorationType
    > {
        const colorContrib = (
            tag: string
        ): [string, vscode.TextEditorDecorationType] => {
            const color = new vscode.ThemeColor('ralsp.' + tag);
            const decor = vscode.window.createTextEditorDecorationType({
                color
            });
            return [tag, decor];
        };

        const decorations: Iterable<
            [string, vscode.TextEditorDecorationType]
        > = [
            colorContrib('comment'),
            colorContrib('string'),
            colorContrib('keyword'),
            colorContrib('keyword.control'),
            colorContrib('keyword.unsafe'),
            colorContrib('function'),
            colorContrib('parameter'),
            colorContrib('constant'),
            colorContrib('type'),
            colorContrib('builtin'),
            colorContrib('text'),
            colorContrib('attribute'),
            colorContrib('literal'),
            colorContrib('macro'),
            colorContrib('variable'),
            colorContrib('field'),
            colorContrib('module')
        ];

        return new Map<string, vscode.TextEditorDecorationType>(decorations);
    }

    private decorations: Map<
        string,
        vscode.TextEditorDecorationType
    > | null = null;

    public removeHighlights() {
        if (this.decorations == null) {
            return;
        }

        // Decorations are removed when the object is disposed
        for (const decoration of this.decorations.values()) {
            decoration.dispose();
        }

        this.decorations = null;
    }

    public setHighlights(editor: vscode.TextEditor, highlights: Decoration[]) {
        // Initialize decorations if necessary
        //
        // Note: decoration objects need to be kept around so we can dispose them
        // if the user disables syntax highlighting
        if (this.decorations == null) {
            this.decorations = Highlighter.initDecorations();
        }

        const byTag: Map<string, vscode.Range[]> = new Map();
        for (const tag of this.decorations.keys()) {
            byTag.set(tag, []);
        }

        for (const d of highlights) {
            if (!byTag.get(d.tag)) {
                continue;
            }
            byTag
                .get(d.tag)!
                .push(Server.client.protocol2CodeConverter.asRange(d.range));
        }

        for (const tag of byTag.keys()) {
            const dec = this.decorations.get(
                tag
            ) as vscode.TextEditorDecorationType;
            const ranges = byTag.get(tag)!;
            editor.setDecorations(dec, ranges);
        }
    }
}
