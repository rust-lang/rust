import * as lc from "vscode-languageclient";
import * as vscode from 'vscode';
import * as ra from './lsp_ext';

import { Ctx, Disposable } from './ctx';
import { sendRequestWithRetry, isRustDocument, RustDocument, RustEditor, sleep } from './util';


export function activateInlayHints(ctx: Ctx) {
    const maybeUpdater = {
        updater: null as null | HintsUpdater,
        async onConfigChange() {
            const anyEnabled = ctx.config.inlayHints.typeHints
                || ctx.config.inlayHints.parameterHints
                || ctx.config.inlayHints.chainingHints;
            const enabled = ctx.config.inlayHints.enable && anyEnabled;

            if (!enabled) return this.dispose();

            await sleep(100);
            if (this.updater) {
                this.updater.syncCacheAndRenderHints();
            } else {
                this.updater = new HintsUpdater(ctx);
            }
        },
        dispose() {
            this.updater?.dispose();
            this.updater = null;
        }
    };

    ctx.pushCleanup(maybeUpdater);

    vscode.workspace.onDidChangeConfiguration(
        maybeUpdater.onConfigChange, maybeUpdater, ctx.subscriptions
    );

    maybeUpdater.onConfigChange();
}

const typeHints = createHintStyle("type");
const paramHints = createHintStyle("parameter");
const chainingHints = createHintStyle("chaining");

function createHintStyle(hintKind: "type" | "parameter" | "chaining") {
    // U+200C is a zero-width non-joiner to prevent the editor from forming a ligature
    // between code and type hints
    const [pos, render] = ({
        type: ["after", (label: string) => `\u{200c}: ${label}`],
        parameter: ["before", (label: string) => `${label}: `],
        chaining: ["after", (label: string) => `\u{200c}: ${label}`],
    } as const)[hintKind];

    const fg = new vscode.ThemeColor(`rust_analyzer.inlayHints.foreground.${hintKind}Hints`);
    const bg = new vscode.ThemeColor(`rust_analyzer.inlayHints.background.${hintKind}Hints`);
    return {
        decorationType: vscode.window.createTextEditorDecorationType({
            [pos]: {
                color: fg,
                backgroundColor: bg,
                fontStyle: "normal",
                fontWeight: "normal",
                textDecoration: ";font-size:smaller",
            },
        }),
        toDecoration(hint: ra.InlayHint, conv: lc.Protocol2CodeConverter): vscode.DecorationOptions {
            return {
                range: conv.asRange(hint.range),
                renderOptions: { [pos]: { contentText: render(hint.label) } }
            };
        }
    };
}

class HintsUpdater implements Disposable {
    private sourceFiles = new Map<string, RustSourceFile>(); // map Uri -> RustSourceFile
    private readonly disposables: Disposable[] = [];

    constructor(private readonly ctx: Ctx) {
        vscode.window.onDidChangeVisibleTextEditors(
            this.onDidChangeVisibleTextEditors,
            this,
            this.disposables
        );

        vscode.workspace.onDidChangeTextDocument(
            this.onDidChangeTextDocument,
            this,
            this.disposables
        );

        // Set up initial cache shape
        ctx.visibleRustEditors.forEach(editor => this.sourceFiles.set(
            editor.document.uri.toString(),
            {
                document: editor.document,
                inlaysRequest: null,
                cachedDecorations: null
            }
        ));

        this.syncCacheAndRenderHints();
    }

    dispose() {
        this.sourceFiles.forEach(file => file.inlaysRequest?.cancel());
        this.ctx.visibleRustEditors.forEach(editor => this.renderDecorations(editor, { param: [], type: [], chaining: [] }));
        this.disposables.forEach(d => d.dispose());
    }

    onDidChangeTextDocument({ contentChanges, document }: vscode.TextDocumentChangeEvent) {
        if (contentChanges.length === 0 || !isRustDocument(document)) return;
        this.syncCacheAndRenderHints();
    }

    syncCacheAndRenderHints() {
        this.sourceFiles.forEach((file, uri) => this.fetchHints(file).then(hints => {
            if (!hints) return;

            file.cachedDecorations = this.hintsToDecorations(hints);

            for (const editor of this.ctx.visibleRustEditors) {
                if (editor.document.uri.toString() === uri) {
                    this.renderDecorations(editor, file.cachedDecorations);
                }
            }
        }));
    }

    onDidChangeVisibleTextEditors() {
        const newSourceFiles = new Map<string, RustSourceFile>();

        // Rerendering all, even up-to-date editors for simplicity
        this.ctx.visibleRustEditors.forEach(async editor => {
            const uri = editor.document.uri.toString();
            const file = this.sourceFiles.get(uri) ?? {
                document: editor.document,
                inlaysRequest: null,
                cachedDecorations: null
            };
            newSourceFiles.set(uri, file);

            // No text documents changed, so we may try to use the cache
            if (!file.cachedDecorations) {
                const hints = await this.fetchHints(file);
                if (!hints) return;

                file.cachedDecorations = this.hintsToDecorations(hints);
            }

            this.renderDecorations(editor, file.cachedDecorations);
        });

        // Cancel requests for no longer visible (disposed) source files
        this.sourceFiles.forEach((file, uri) => {
            if (!newSourceFiles.has(uri)) file.inlaysRequest?.cancel();
        });

        this.sourceFiles = newSourceFiles;
    }

    private renderDecorations(editor: RustEditor, decorations: InlaysDecorations) {
        editor.setDecorations(typeHints.decorationType, decorations.type);
        editor.setDecorations(paramHints.decorationType, decorations.param);
        editor.setDecorations(chainingHints.decorationType, decorations.chaining);
    }

    private hintsToDecorations(hints: ra.InlayHint[]): InlaysDecorations {
        const decorations: InlaysDecorations = { type: [], param: [], chaining: [] };
        const conv = this.ctx.client.protocol2CodeConverter;

        for (const hint of hints) {
            switch (hint.kind) {
                case ra.InlayHint.Kind.TypeHint: {
                    decorations.type.push(typeHints.toDecoration(hint, conv));
                    continue;
                }
                case ra.InlayHint.Kind.ParamHint: {
                    decorations.param.push(paramHints.toDecoration(hint, conv));
                    continue;
                }
                case ra.InlayHint.Kind.ChainingHint: {
                    decorations.chaining.push(chainingHints.toDecoration(hint, conv));
                    continue;
                }
            }
        }
        return decorations;
    }

    private async fetchHints(file: RustSourceFile): Promise<null | ra.InlayHint[]> {
        file.inlaysRequest?.cancel();

        const tokenSource = new vscode.CancellationTokenSource();
        file.inlaysRequest = tokenSource;

        const request = { textDocument: { uri: file.document.uri.toString() } };

        return sendRequestWithRetry(this.ctx.client, ra.inlayHints, request, tokenSource.token)
            .catch(_ => null)
            .finally(() => {
                if (file.inlaysRequest === tokenSource) {
                    file.inlaysRequest = null;
                }
            });
    }
}

interface InlaysDecorations {
    type: vscode.DecorationOptions[];
    param: vscode.DecorationOptions[];
    chaining: vscode.DecorationOptions[];
}

interface RustSourceFile {
    /**
     * Source of the token to cancel in-flight inlay hints request if any.
     */
    inlaysRequest: null | vscode.CancellationTokenSource;
    /**
     * Last applied decorations.
     */
    cachedDecorations: null | InlaysDecorations;

    document: RustDocument;
}
