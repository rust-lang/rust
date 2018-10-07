import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient'

import { Server } from './server';

export interface Decoration {
    range: lc.Range,
    tag: string,
}

export class Highlighter {
    private decorations: { [index: string]: vscode.TextEditorDecorationType };
    constructor() {
        this.decorations = {};
    }

    removeHighlights() {
        for (let tag in this.decorations) {
            this.decorations[tag].dispose();
        }

        this.decorations = {};
    }

    setHighlights(
        editor: vscode.TextEditor,
        highlights: Array<Decoration>
    ) {
        // Initialize decorations if necessary
        //
        // Note: decoration objects need to be kept around so we can dispose them
        // if the user disables syntax highlighting
        if (Object.keys(this.decorations).length === 0) {
            this.initDecorations();
        }

        let byTag: Map<string, vscode.Range[]> = new Map()
        for (let tag in this.decorations) {
            byTag.set(tag, [])
        }

        for (let d of highlights) {
            if (!byTag.get(d.tag)) {
                console.log(`unknown tag ${d.tag}`)
                continue
            }
            byTag.get(d.tag)!.push(
                Server.client.protocol2CodeConverter.asRange(d.range)
            )
        }

        for (let tag of byTag.keys()) {
            let dec: vscode.TextEditorDecorationType = this.decorations[tag]
            let ranges = byTag.get(tag)!
            editor.setDecorations(dec, ranges)
        }
    }

    private initDecorations() {
        const decor = (obj: any) => vscode.window.createTextEditorDecorationType({ color: obj })
        this.decorations = {
            background: decor("#3F3F3F"),
            error: vscode.window.createTextEditorDecorationType({
                borderColor: "red",
                borderStyle: "none none dashed none",
            }),
            comment: decor("#7F9F7F"),
            string: decor("#CC9393"),
            keyword: decor("#F0DFAF"),
            function: decor("#93E0E3"),
            parameter: decor("#94BFF3"),
            builtin: decor("#DD6718"),
            text: decor("#DCDCCC"),
            attribute: decor("#BFEBBF"),
            literal: decor("#DFAF8F"),
        }
    }
}
