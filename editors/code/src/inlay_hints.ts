import * as vscode from 'vscode';
import * as ra from './lsp_ext';

import { Ctx, Disposable } from './ctx';
import { sendRequestWithRetry, isRustDocument } from './util';

export function activateInlayHints(ctx: Ctx) {
    const maybeUpdater = {
        hintsProvider: null as Disposable | null,
        updateHintsEventEmitter: new vscode.EventEmitter<void>(),

        async onConfigChange() {
            this.dispose();

            const anyEnabled = ctx.config.inlayHints.typeHints
                || ctx.config.inlayHints.parameterHints
                || ctx.config.inlayHints.chainingHints;
            const enabled = ctx.config.inlayHints.enable && anyEnabled;
            if (!enabled) return;

            const event = this.updateHintsEventEmitter.event;
            this.hintsProvider = vscode.languages.registerInlayHintsProvider({ scheme: 'file', language: 'rust' }, new class implements vscode.InlayHintsProvider {
                onDidChangeInlayHints = event;
                async provideInlayHints(document: vscode.TextDocument, range: vscode.Range, token: vscode.CancellationToken): Promise<vscode.InlayHint[]> {
                    const request = { textDocument: { uri: document.uri.toString() }, range: { start: range.start, end: range.end } };
                    const hints = await sendRequestWithRetry(ctx.client, ra.inlayHints, request, token).catch(_ => null);
                    if (hints == null) {
                        return [];
                    } else {
                        return hints;
                    }
                }
            });
        },

        onDidChangeTextDocument({ contentChanges, document }: vscode.TextDocumentChangeEvent) {
            if (contentChanges.length === 0 || !isRustDocument(document)) return;
            this.updateHintsEventEmitter.fire();
        },

        dispose() {
            this.hintsProvider?.dispose();
            this.hintsProvider = null;
            this.updateHintsEventEmitter.dispose();
        },
    };

    ctx.pushCleanup(maybeUpdater);

    vscode.workspace.onDidChangeConfiguration(maybeUpdater.onConfigChange, maybeUpdater, ctx.subscriptions);
    vscode.workspace.onDidChangeTextDocument(maybeUpdater.onDidChangeTextDocument, maybeUpdater, ctx.subscriptions);

    maybeUpdater.onConfigChange().catch(console.error);
}
