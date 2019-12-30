import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Ctx } from './ctx';

export function activateInlayHints(ctx: Ctx) {
    const hintsUpdater = new HintsUpdater(ctx);
    vscode.window.onDidChangeVisibleTextEditors(async _ => {
        await hintsUpdater.refresh();
    }, ctx.subscriptions);

    vscode.workspace.onDidChangeTextDocument(async e => {
        if (e.contentChanges.length === 0) return;
        if (e.document.languageId !== 'rust') return;
        await hintsUpdater.refresh();
    }, ctx.subscriptions);

    vscode.workspace.onDidChangeConfiguration(_ => {
        hintsUpdater.setEnabled(ctx.config.displayInlayHints);
    }, ctx.subscriptions);

    // XXX: don't await here;
    // Who knows what happens if an exception is thrown here...
    hintsUpdater.refresh();
}

interface InlayHintsParams {
    textDocument: lc.TextDocumentIdentifier;
}

interface InlayHint {
    range: vscode.Range;
    kind: string;
    label: string;
}

const typeHintDecorationType = vscode.window.createTextEditorDecorationType({
    after: {
        color: new vscode.ThemeColor('ralsp.inlayHint'),
    },
});

class HintsUpdater {
    private ctx: Ctx;
    private enabled = true;

    constructor(ctx: Ctx) {
        this.ctx = ctx;
    }

    async setEnabled(enabled: boolean) {
        if (this.enabled == enabled) return;
        this.enabled = enabled;

        if (this.enabled) {
            await this.refresh();
        } else {
            this.allEditors.forEach(it => this.setDecorations(it, []));
        }
    }

    async refresh() {
        if (!this.enabled) return;
        const promises = this.allEditors.map(it => this.refreshEditor(it));
        await Promise.all(promises);
    }

    private async refreshEditor(editor: vscode.TextEditor): Promise<void> {
        const newHints = await this.queryHints(editor.document.uri.toString());
        const newDecorations = (newHints ? newHints : []).map(hint => ({
            range: hint.range,
            renderOptions: {
                after: {
                    contentText: `: ${hint.label}`,
                },
            },
        }));
        this.setDecorations(editor, newDecorations);
    }

    private get allEditors(): vscode.TextEditor[] {
        return vscode.window.visibleTextEditors.filter(
            editor => editor.document.languageId === 'rust',
        );
    }

    private setDecorations(
        editor: vscode.TextEditor,
        decorations: vscode.DecorationOptions[],
    ) {
        editor.setDecorations(
            typeHintDecorationType,
            this.enabled ? decorations : [],
        );
    }

    private async queryHints(documentUri: string): Promise<InlayHint[] | null> {
        const request: InlayHintsParams = {
            textDocument: { uri: documentUri },
        };
        return this.ctx.sendRequestWithRetry<InlayHint[] | null>(
            'rust-analyzer/inlayHints',
            request,
        );
    }
}
