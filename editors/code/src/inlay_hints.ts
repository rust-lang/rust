import * as lc from "vscode-languageclient";
import * as vscode from 'vscode';
import * as ra from './rust-analyzer-api';

import { Ctx } from './ctx';
import { sendRequestWithRetry, assert } from './util';

export function activateInlayHints(ctx: Ctx) {
    const hintsUpdater = new HintsUpdater(ctx.client);

    vscode.window.onDidChangeVisibleTextEditors(
        visibleEditors => hintsUpdater.refreshVisibleRustEditors(
            visibleEditors.filter(isRustTextEditor)
        ),
        null,
        ctx.subscriptions
    );

    vscode.workspace.onDidChangeTextDocument(
        ({ contentChanges, document }) => {
            if (contentChanges.length === 0) return;
            if (!isRustTextDocument(document)) return;

            hintsUpdater.refreshRustDocument(document);
        },
        null,
        ctx.subscriptions
    );

    vscode.workspace.onDidChangeConfiguration(
        async _ => {
            // FIXME: ctx.config may have not been refreshed at this point of time, i.e.
            // it's on onDidChangeConfiguration() handler may've not executed yet
            // (order of invokation is unspecified)
            // To fix this we should expose an event emitter from our `Config` itself.
            await hintsUpdater.setEnabled(ctx.config.displayInlayHints);
        },
        null,
        ctx.subscriptions
    );

    ctx.pushCleanup({
        dispose() {
            hintsUpdater.clearHints();
        }
    });

    hintsUpdater.setEnabled(ctx.config.displayInlayHints);
}


const typeHints = {
    decorationType: vscode.window.createTextEditorDecorationType({
        after: {
            color: new vscode.ThemeColor('rust_analyzer.inlayHint'),
            fontStyle: "normal",
        }
    }),

    toDecoration(hint: ra.InlayHint.TypeHint, conv: lc.Protocol2CodeConverter): vscode.DecorationOptions {
        return {
            range: conv.asRange(hint.range),
            renderOptions: { after: { contentText: `: ${hint.label}` } }
        };
    }
};

const paramHints = {
    decorationType: vscode.window.createTextEditorDecorationType({
        before: {
            color: new vscode.ThemeColor('rust_analyzer.inlayHint'),
            fontStyle: "normal",
        }
    }),

    toDecoration(hint: ra.InlayHint.ParamHint, conv: lc.Protocol2CodeConverter): vscode.DecorationOptions {
        return {
            range: conv.asRange(hint.range),
            renderOptions: { before: { contentText: `${hint.label}: ` } }
        };
    }
};

class HintsUpdater {
    private sourceFiles = new RustSourceFiles();
    private enabled = false;

    constructor(readonly client: lc.LanguageClient) { }

    setEnabled(enabled: boolean) {
        if (this.enabled === enabled) return;
        this.enabled = enabled;

        if (this.enabled) {
            this.refreshVisibleRustEditors(vscode.window.visibleTextEditors.filter(isRustTextEditor));
        } else {
            this.clearHints();
        }
    }

    clearHints() {
        for (const file of this.sourceFiles) {
            file.inlaysRequest?.cancel();
            this.renderHints(file, []);
        }
    }

    private renderHints(file: RustSourceFile, hints: ra.InlayHint[]) {
        file.renderHints(hints, this.client.protocol2CodeConverter);
    }

    refreshRustDocument(document: RustTextDocument) {
        if (!this.enabled) return;

        const file = this.sourceFiles.getSourceFile(document.uri.toString());

        assert(!!file, "Document must be opened in some text editor!");

        void file.fetchAndRenderHints(this.client);
    }

    refreshVisibleRustEditors(visibleEditors: RustTextEditor[]) {
        if (!this.enabled) return;

        const visibleSourceFiles = this.sourceFiles.drainEditors(visibleEditors);

        // Cancel requests for source files whose editors were disposed (leftovers after drain).
        for (const { inlaysRequest } of this.sourceFiles) inlaysRequest?.cancel();

        this.sourceFiles = visibleSourceFiles;

        for (const file of this.sourceFiles) {
            if (!file.rerenderHints()) {
                void file.fetchAndRenderHints(this.client);
            }
        }
    }
}


/**
 * This class encapsulates a map of file uris to respective inlay hints
 * request cancellation token source (cts) and an array of editors.
 * E.g.
 * ```
 * {
 *    file1.rs -> (cts, (typeDecor, paramDecor), [editor1, editor2])
 *                  ^-- there is a cts to cancel the in-flight request
 *    file2.rs -> (cts, null, [editor3])
 *                       ^-- no decorations are applied to this source file yet
 *    file3.rs -> (null, (typeDecor, paramDecor), [editor4])
 * }                ^-- there is no inflight request
 * ```
 *
 * Invariants: each stored source file has at least 1 editor.
 */
class RustSourceFiles {
    private files = new Map<string, RustSourceFile>();

    /**
     * Removes `editors` from `this` source files and puts them into a returned
     * source files object. cts and decorations are moved to the returned source files.
     */
    drainEditors(editors: RustTextEditor[]): RustSourceFiles {
        const result = new RustSourceFiles;

        for (const editor of editors) {
            const oldFile = this.removeEditor(editor);
            const newFile = result.addEditor(editor);

            if (oldFile) newFile.stealCacheFrom(oldFile);
        }

        return result;
    }

    /**
     * Remove the editor and if it was the only editor for a source file,
     * the source file is removed altogether.
     *
     * @returns A reference to the source file for this editor or
     *          null if no such source file was not found.
     */
    private removeEditor(editor: RustTextEditor): null | RustSourceFile {
        const uri = editor.document.uri.toString();

        const file = this.files.get(uri);
        if (!file) return null;

        const editorIndex = file.editors.findIndex(suspect => areEditorsEqual(suspect, editor));

        if (editorIndex >= 0) {
            file.editors.splice(editorIndex, 1);

            if (file.editors.length === 0) this.files.delete(uri);
        }

        return file;
    }

    /**
     * @returns A reference to an existing source file or newly created one for the editor.
     */
    private addEditor(editor: RustTextEditor): RustSourceFile {
        const uri = editor.document.uri.toString();
        const file = this.files.get(uri);

        if (!file) {
            const newFile = new RustSourceFile([editor]);
            this.files.set(uri, newFile);
            return newFile;
        }

        if (!file.editors.find(suspect => areEditorsEqual(suspect, editor))) {
            file.editors.push(editor);
        }
        return file;
    }

    getSourceFile(uri: string): undefined | RustSourceFile {
        return this.files.get(uri);
    }

    [Symbol.iterator](): IterableIterator<RustSourceFile> {
        return this.files.values();
    }
}
class RustSourceFile {
    constructor(
        /**
         * Editors for this source file (one text document may be opened in multiple editors).
         * We keep this just an array, because most of the time we have 1 editor for 1 source file.
         */
        readonly editors: RustTextEditor[],
        /**
         * Source of the token to cancel in-flight inlay hints request if any.
         */
        public inlaysRequest: null | vscode.CancellationTokenSource = null,

        public decorations: null | {
            type: vscode.DecorationOptions[];
            param: vscode.DecorationOptions[];
        } = null
    ) { }

    stealCacheFrom(other: RustSourceFile) {
        if (other.inlaysRequest) this.inlaysRequest = other.inlaysRequest;
        if (other.decorations) this.decorations = other.decorations;

        other.inlaysRequest = null;
        other.decorations = null;
    }

    rerenderHints(): boolean {
        if (!this.decorations) return false;

        for (const editor of this.editors) {
            editor.setDecorations(typeHints.decorationType, this.decorations.type);
            editor.setDecorations(paramHints.decorationType, this.decorations.param);
        }
        return true;
    }

    renderHints(hints: ra.InlayHint[], conv: lc.Protocol2CodeConverter) {
        this.decorations = { type: [], param: [] };

        for (const hint of hints) {
            switch (hint.kind) {
                case ra.InlayHint.Kind.TypeHint: {
                    this.decorations.type.push(typeHints.toDecoration(hint, conv));
                    continue;
                }
                case ra.InlayHint.Kind.ParamHint: {
                    this.decorations.param.push(paramHints.toDecoration(hint, conv));
                    continue;
                }
            }
        }
        this.rerenderHints();
    }

    async fetchAndRenderHints(client: lc.LanguageClient): Promise<void> {
        this.inlaysRequest?.cancel();

        const tokenSource = new vscode.CancellationTokenSource();
        this.inlaysRequest = tokenSource;

        const request = { textDocument: { uri: this.editors[0].document.uri.toString() } };

        try {
            const hints = await sendRequestWithRetry(client, ra.inlayHints, request, tokenSource.token);
            this.renderHints(hints, client.protocol2CodeConverter);
        } catch {
            /* ignore */
        } finally {
            if (this.inlaysRequest === tokenSource) {
                this.inlaysRequest = null;
            }
        }
    }
}

type RustTextDocument = vscode.TextDocument & { languageId: "rust" };
type RustTextEditor = vscode.TextEditor & { document: RustTextDocument; id: string };

function areEditorsEqual(a: RustTextEditor, b: RustTextEditor): boolean {
    return a.id === b.id;
}

function isRustTextEditor(suspect: vscode.TextEditor & { id?: unknown }): suspect is RustTextEditor {
    // Dirty hack, we need to access private vscode editor id,
    // see https://github.com/microsoft/vscode/issues/91788
    assert(
        typeof suspect.id === "string",
        "Private text editor id is no longer available, please update the workaround!"
    );

    return isRustTextDocument(suspect.document);
}

function isRustTextDocument(suspect: vscode.TextDocument): suspect is RustTextDocument {
    return suspect.languageId === "rust";
}
