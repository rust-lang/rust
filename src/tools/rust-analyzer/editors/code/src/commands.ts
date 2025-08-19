import * as vscode from "vscode";
import * as lc from "vscode-languageclient";
import * as ra from "./lsp_ext";
import * as path from "path";

import type { Ctx, Cmd, CtxInit } from "./ctx";
import {
    applySnippetWorkspaceEdit,
    applySnippetTextEdits,
    type SnippetTextDocumentEdit,
} from "./snippets";
import {
    type RunnableQuickPick,
    selectRunnable,
    createTaskFromRunnable,
    createCargoArgs,
} from "./run";
import {
    isRustDocument,
    isCargoRunnableArgs,
    isCargoTomlDocument,
    sleep,
    isRustEditor,
    type RustEditor,
    type RustDocument,
    unwrapUndefinable,
} from "./util";
import { startDebugSession, makeDebugConfig } from "./debug";
import type { LanguageClient } from "vscode-languageclient/node";
import { HOVER_REFERENCE_COMMAND } from "./client";
import type { DependencyId } from "./dependencies_provider";
import { log } from "./util";
import type { SyntaxElement } from "./syntax_tree_provider";

export * from "./run";

export function analyzerStatus(ctx: CtxInit): Cmd {
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse("rust-analyzer-status://status");
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();

        async provideTextDocumentContent(_uri: vscode.Uri): Promise<string> {
            if (!vscode.window.activeTextEditor) return "";
            const client = ctx.client;

            const params: ra.AnalyzerStatusParams = {};
            const doc = ctx.activeRustEditor?.document;
            if (doc != null) {
                params.textDocument = client.code2ProtocolConverter.asTextDocumentIdentifier(doc);
            }
            return await client.sendRequest(ra.analyzerStatus, params);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-status", tdcp),
    );

    return async () => {
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        tdcp.eventEmitter.fire(tdcp.uri);
        void (await vscode.window.showTextDocument(document, {
            viewColumn: vscode.ViewColumn.Two,
            preserveFocus: true,
        }));
    };
}

export function memoryUsage(ctx: CtxInit): Cmd {
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse("rust-analyzer-memory://memory");
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();

        provideTextDocumentContent(_uri: vscode.Uri): vscode.ProviderResult<string> {
            if (!vscode.window.activeTextEditor) return "";

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            return ctx.client.sendRequest(ra.memoryUsage).then((mem: any) => {
                return "Per-query memory usage:\n" + mem + "\n(note: database has been cleared)";
            });
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-memory", tdcp),
    );

    return async () => {
        tdcp.eventEmitter.fire(tdcp.uri);
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        return vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true);
    };
}

export function triggerParameterHints(_: CtxInit): Cmd {
    return async () => {
        const parameterHintsEnabled = vscode.workspace
            .getConfiguration("editor")
            .get<boolean>("parameterHints.enabled");

        if (parameterHintsEnabled) {
            await vscode.commands.executeCommand("editor.action.triggerParameterHints");
        }
    };
}

export function rename(_: CtxInit): Cmd {
    return async () => {
        await vscode.commands.executeCommand("editor.action.rename");
    };
}

export function openLogs(ctx: CtxInit): Cmd {
    return async () => {
        if (ctx.client.outputChannel) {
            ctx.client.outputChannel.show();
        }
    };
}

export function matchingBrace(ctx: CtxInit): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        const client = ctx.client;

        const response = await client.sendRequest(ra.matchingBrace, {
            textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(editor.document),
            positions: editor.selections.map((s) =>
                client.code2ProtocolConverter.asPosition(s.active),
            ),
        });
        editor.selections = editor.selections.map((sel, idx) => {
            const position = unwrapUndefinable(response[idx]);
            const active = client.protocol2CodeConverter.asPosition(position);
            const anchor = sel.isEmpty ? active : sel.anchor;
            return new vscode.Selection(anchor, active);
        });
        editor.revealRange(editor.selection);
    };
}

export function joinLines(ctx: CtxInit): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        const client = ctx.client;

        const items: lc.TextEdit[] = await client.sendRequest(ra.joinLines, {
            ranges: editor.selections.map((it) => client.code2ProtocolConverter.asRange(it)),
            textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(editor.document),
        });
        const textEdits = await client.protocol2CodeConverter.asTextEdits(items);
        await editor.edit((builder) => {
            textEdits.forEach((edit: vscode.TextEdit) => {
                builder.replace(edit.range, edit.newText);
            });
        });
    };
}

export function moveItemUp(ctx: CtxInit): Cmd {
    return moveItem(ctx, "Up");
}

export function moveItemDown(ctx: CtxInit): Cmd {
    return moveItem(ctx, "Down");
}

export function moveItem(ctx: CtxInit, direction: ra.Direction): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;
        const client = ctx.client;

        const lcEdits = await client.sendRequest(ra.moveItem, {
            range: client.code2ProtocolConverter.asRange(editor.selection),
            textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(editor.document),
            direction,
        });

        if (!lcEdits) return;

        const edits = await client.protocol2CodeConverter.asTextEdits(lcEdits);
        await applySnippetTextEdits(editor, edits);
    };
}

export function onEnter(ctx: CtxInit): Cmd {
    async function handleKeypress() {
        const editor = ctx.activeRustEditor;

        if (!editor) return false;

        const client = ctx.client;
        const lcEdits = await client
            .sendRequest(ra.onEnter, {
                textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(
                    editor.document,
                ),
                position: client.code2ProtocolConverter.asPosition(editor.selection.active),
            })
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .catch((_error: any) => {
                // client.handleFailedRequest(OnEnterRequest.type, error, null);
                return null;
            });
        if (!lcEdits) return false;

        const edits = await client.protocol2CodeConverter.asTextEdits(lcEdits);
        await applySnippetTextEdits(editor, edits);
        return true;
    }

    return async () => {
        if (await handleKeypress()) return;

        await vscode.commands.executeCommand("default:type", { text: "\n" });
    };
}

export function parentModule(ctx: CtxInit): Cmd {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        if (!(isRustDocument(editor.document) || isCargoTomlDocument(editor.document))) return;

        const client = ctx.client;

        const locations = await client.sendRequest(ra.parentModule, {
            textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(editor.document),
            position: client.code2ProtocolConverter.asPosition(editor.selection.active),
        });
        if (!locations) return;

        if (locations.length === 1) {
            const loc = unwrapUndefinable(locations[0]);

            const uri = client.protocol2CodeConverter.asUri(loc.targetUri);
            const range = client.protocol2CodeConverter.asRange(loc.targetRange);

            const doc = await vscode.workspace.openTextDocument(uri);
            const e = await vscode.window.showTextDocument(doc);
            e.selection = new vscode.Selection(range.start, range.start);
            e.revealRange(range, vscode.TextEditorRevealType.InCenter);
        } else {
            const uri = editor.document.uri.toString();
            const position = client.code2ProtocolConverter.asPosition(editor.selection.active);
            await showReferencesImpl(
                client,
                uri,
                position,
                locations.map((loc) => lc.Location.create(loc.targetUri, loc.targetRange)),
            );
        }
    };
}

export function childModules(ctx: CtxInit): Cmd {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        if (!(isRustDocument(editor.document) || isCargoTomlDocument(editor.document))) return;

        const client = ctx.client;

        const locations = await client.sendRequest(ra.childModules, {
            textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(editor.document),
            position: client.code2ProtocolConverter.asPosition(editor.selection.active),
        });
        if (!locations) return;

        if (locations.length === 1) {
            const loc = unwrapUndefinable(locations[0]);

            const uri = client.protocol2CodeConverter.asUri(loc.targetUri);
            const range = client.protocol2CodeConverter.asRange(loc.targetRange);

            const doc = await vscode.workspace.openTextDocument(uri);
            const e = await vscode.window.showTextDocument(doc);
            e.selection = new vscode.Selection(range.start, range.start);
            e.revealRange(range, vscode.TextEditorRevealType.InCenter);
        } else {
            const uri = editor.document.uri.toString();
            const position = client.code2ProtocolConverter.asPosition(editor.selection.active);
            await showReferencesImpl(
                client,
                uri,
                position,
                locations.map((loc) => lc.Location.create(loc.targetUri, loc.targetRange)),
            );
        }
    };
}

export function openCargoToml(ctx: CtxInit): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        const client = ctx.client;
        const response = await client.sendRequest(ra.openCargoToml, {
            textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(editor.document),
        });
        if (!response) return;

        const uri = client.protocol2CodeConverter.asUri(response.uri);
        const range = client.protocol2CodeConverter.asRange(response.range);

        const doc = await vscode.workspace.openTextDocument(uri);
        const e = await vscode.window.showTextDocument(doc);
        e.selection = new vscode.Selection(range.start, range.start);
        e.revealRange(range, vscode.TextEditorRevealType.InCenter);
    };
}

export function revealDependency(ctx: CtxInit): Cmd {
    return async (editor: RustEditor) => {
        if (!ctx.dependenciesProvider?.isInitialized()) {
            return;
        }
        const documentPath = editor.document.uri.fsPath;
        const dep = ctx.dependenciesProvider?.getDependency(documentPath);
        if (dep) {
            await ctx.dependencyTreeView?.reveal(dep, { select: true, expand: true });
        } else {
            await revealParentChain(editor.document, ctx);
        }
    };
}

/**
 * This function calculates the parent chain of a given file until it reaches it crate root contained in ctx.dependencies.
 * This is need because the TreeView is Lazy, so at first it only has the root dependencies: For example if we have the following crates:
 * - core
 * - alloc
 * - std
 *
 * if I want to reveal alloc/src/str.rs, I have to:

 * 1. reveal every children of alloc
 * - core
 * - alloc\
 * &emsp;|-beches\
 * &emsp;|-src\
 * &emsp;|- ...
 * - std
 * 2. reveal every children of src:
 * core
 * alloc\
 * &emsp;|-beches\
 * &emsp;|-src\
 * &emsp;&emsp;|- lib.rs\
 * &emsp;&emsp;|- str.rs <------- FOUND IT!\
 * &emsp;&emsp;|- ...\
 * &emsp;|- ...\
 * std
 */
async function revealParentChain(document: RustDocument, ctx: CtxInit) {
    let documentPath = document.uri.fsPath;
    const maxDepth = documentPath.split(path.sep).length - 1;
    const parentChain: DependencyId[] = [{ id: documentPath.toLowerCase() }];
    do {
        documentPath = path.dirname(documentPath);
        parentChain.push({ id: documentPath.toLowerCase() });
        if (parentChain.length >= maxDepth) {
            // this is an odd case that can happen when we change a crate version but we'd still have
            // a open file referencing the old version
            return;
        }
    } while (!ctx.dependenciesProvider?.contains(documentPath));
    parentChain.reverse();
    for (const idx in parentChain) {
        const treeView = ctx.dependencyTreeView;
        if (!treeView) {
            continue;
        }

        const dependency = unwrapUndefinable(parentChain[idx]);
        await treeView.reveal(dependency, { select: true, expand: true });
    }
}

export async function execRevealDependency(e: RustEditor): Promise<void> {
    await vscode.commands.executeCommand("rust-analyzer.revealDependency", e);
}

export function syntaxTreeReveal(): Cmd {
    return async (element: SyntaxElement) => {
        const activeEditor = vscode.window.activeTextEditor;

        if (activeEditor !== undefined) {
            const newSelection = new vscode.Selection(element.range.start, element.range.end);

            activeEditor.selection = newSelection;
            activeEditor.revealRange(newSelection);
        }
    };
}

function elementToString(
    activeDocument: vscode.TextDocument,
    element: SyntaxElement,
    depth: number = 0,
): string {
    let result = "  ".repeat(depth);
    const offsets = element.inner?.offsets ?? element.offsets;

    result += `${element.kind}@${offsets.start}..${offsets.end}`;

    if (element.type === "Token") {
        const text = activeDocument.getText(element.range).replaceAll("\r\n", "\n");
        // JSON.stringify quotes and escapes the string for us.
        result += ` ${JSON.stringify(text)}\n`;
    } else {
        result += "\n";
        for (const child of element.children) {
            result += elementToString(activeDocument, child, depth + 1);
        }
    }

    return result;
}

export function syntaxTreeCopy(): Cmd {
    return async (element: SyntaxElement) => {
        const activeDocument = vscode.window.activeTextEditor?.document;
        if (!activeDocument) {
            return;
        }

        const result = elementToString(activeDocument, element);
        await vscode.env.clipboard.writeText(result);
    };
}

export function syntaxTreeHideWhitespace(ctx: CtxInit): Cmd {
    return async () => {
        if (ctx.syntaxTreeProvider !== undefined) {
            await ctx.syntaxTreeProvider.toggleWhitespace();
        }
    };
}

export function syntaxTreeShowWhitespace(ctx: CtxInit): Cmd {
    return async () => {
        if (ctx.syntaxTreeProvider !== undefined) {
            await ctx.syntaxTreeProvider.toggleWhitespace();
        }
    };
}

export function ssr(ctx: CtxInit): Cmd {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const client = ctx.client;

        const position = editor.selection.active;
        const selections = editor.selections;
        const textDocument = client.code2ProtocolConverter.asTextDocumentIdentifier(
            editor.document,
        );

        const options: vscode.InputBoxOptions = {
            value: "() ==>> ()",
            prompt: "Enter request, for example 'Foo($a) ==>> Foo::new($a)' ",
            validateInput: async (x: string) => {
                try {
                    await client.sendRequest(ra.ssr, {
                        query: x,
                        parseOnly: true,
                        textDocument,
                        position,
                        selections,
                    });
                } catch (e) {
                    return String(e);
                }
                return null;
            },
        };
        const request = await vscode.window.showInputBox(options);
        if (!request) return;

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: "Structured search replace in progress...",
                cancellable: false,
            },
            async (_progress, token) => {
                const edit = await client.sendRequest(ra.ssr, {
                    query: request,
                    parseOnly: false,
                    textDocument,
                    position,
                    selections,
                });

                await vscode.workspace.applyEdit(
                    await client.protocol2CodeConverter.asWorkspaceEdit(edit, token),
                );
            },
        );
    };
}

export function serverVersion(ctx: CtxInit): Cmd {
    return async () => {
        if (!ctx.serverPath) {
            void vscode.window.showWarningMessage(`rust-analyzer server is not running`);
            return;
        }
        void vscode.window.showInformationMessage(
            `rust-analyzer version: ${ctx.serverVersion} [${ctx.serverPath}]`,
        );
    };
}

function viewHirOrMir(ctx: CtxInit, xir: "hir" | "mir"): Cmd {
    const viewXir = xir === "hir" ? "viewHir" : "viewMir";
    const requestType = xir === "hir" ? ra.viewHir : ra.viewMir;
    const uri = `rust-analyzer-${xir}://${viewXir}/${xir}.rs`;
    const scheme = `rust-analyzer-${xir}`;
    return viewFileUsingTextDocumentContentProvider(ctx, requestType, uri, scheme, true);
}

function viewFileUsingTextDocumentContentProvider(
    ctx: CtxInit,
    requestType: lc.RequestType<lc.TextDocumentPositionParams, string, void>,
    uri: string,
    scheme: string,
    shouldUpdate: boolean,
): Cmd {
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse(uri);
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();
        constructor() {
            vscode.workspace.onDidChangeTextDocument(
                this.onDidChangeTextDocument,
                this,
                ctx.subscriptions,
            );
            vscode.window.onDidChangeActiveTextEditor(
                this.onDidChangeActiveTextEditor,
                this,
                ctx.subscriptions,
            );
        }

        private onDidChangeTextDocument(event: vscode.TextDocumentChangeEvent) {
            if (isRustDocument(event.document) && shouldUpdate) {
                // We need to order this after language server updates, but there's no API for that.
                // Hence, good old sleep().
                void sleep(10).then(() => this.eventEmitter.fire(this.uri));
            }
        }

        private onDidChangeActiveTextEditor(editor: vscode.TextEditor | undefined) {
            if (editor && isRustEditor(editor) && shouldUpdate) {
                this.eventEmitter.fire(this.uri);
            }
        }

        async provideTextDocumentContent(
            _uri: vscode.Uri,
            ct: vscode.CancellationToken,
        ): Promise<string> {
            const rustEditor = ctx.activeRustEditor;
            if (!rustEditor) return "";

            const client = ctx.client;
            const params = {
                textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(
                    rustEditor.document,
                ),
                position: client.code2ProtocolConverter.asPosition(rustEditor.selection.active),
            };
            return client.sendRequest(requestType, params, ct);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(vscode.workspace.registerTextDocumentContentProvider(scheme, tdcp));

    return async () => {
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        tdcp.eventEmitter.fire(tdcp.uri);
        void (await vscode.window.showTextDocument(document, {
            viewColumn: vscode.ViewColumn.Two,
            preserveFocus: true,
        }));
    };
}

// Opens the virtual file that will show the HIR of the function containing the cursor position
//
// The contents of the file come from the `TextDocumentContentProvider`
export function viewHir(ctx: CtxInit): Cmd {
    return viewHirOrMir(ctx, "hir");
}

// Opens the virtual file that will show the MIR of the function containing the cursor position
//
// The contents of the file come from the `TextDocumentContentProvider`
export function viewMir(ctx: CtxInit): Cmd {
    return viewHirOrMir(ctx, "mir");
}

// Opens the virtual file that will show the MIR of the function containing the cursor position
//
// The contents of the file come from the `TextDocumentContentProvider`
export function interpretFunction(ctx: CtxInit): Cmd {
    const uri = `rust-analyzer-interpret-function://interpretFunction/result.log`;
    return viewFileUsingTextDocumentContentProvider(
        ctx,
        ra.interpretFunction,
        uri,
        `rust-analyzer-interpret-function`,
        false,
    );
}

export function viewFileText(ctx: CtxInit): Cmd {
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse("rust-analyzer-file-text://viewFileText/file.rs");
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();
        constructor() {
            vscode.workspace.onDidChangeTextDocument(
                this.onDidChangeTextDocument,
                this,
                ctx.subscriptions,
            );
            vscode.window.onDidChangeActiveTextEditor(
                this.onDidChangeActiveTextEditor,
                this,
                ctx.subscriptions,
            );
        }

        private onDidChangeTextDocument(event: vscode.TextDocumentChangeEvent) {
            if (isRustDocument(event.document)) {
                // We need to order this after language server updates, but there's no API for that.
                // Hence, good old sleep().
                void sleep(10).then(() => this.eventEmitter.fire(this.uri));
            }
        }

        private onDidChangeActiveTextEditor(editor: vscode.TextEditor | undefined) {
            if (editor && isRustEditor(editor)) {
                this.eventEmitter.fire(this.uri);
            }
        }

        async provideTextDocumentContent(
            _uri: vscode.Uri,
            ct: vscode.CancellationToken,
        ): Promise<string> {
            const rustEditor = ctx.activeRustEditor;
            if (!rustEditor) return "";
            const client = ctx.client;

            const params = client.code2ProtocolConverter.asTextDocumentIdentifier(
                rustEditor.document,
            );
            return client.sendRequest(ra.viewFileText, params, ct);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-file-text", tdcp),
    );

    return async () => {
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        tdcp.eventEmitter.fire(tdcp.uri);
        void (await vscode.window.showTextDocument(document, {
            viewColumn: vscode.ViewColumn.Two,
            preserveFocus: true,
        }));
    };
}

export function viewItemTree(ctx: CtxInit): Cmd {
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse("rust-analyzer-item-tree://viewItemTree/itemtree.rs");
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();
        constructor() {
            vscode.workspace.onDidChangeTextDocument(
                this.onDidChangeTextDocument,
                this,
                ctx.subscriptions,
            );
            vscode.window.onDidChangeActiveTextEditor(
                this.onDidChangeActiveTextEditor,
                this,
                ctx.subscriptions,
            );
        }

        private onDidChangeTextDocument(event: vscode.TextDocumentChangeEvent) {
            if (isRustDocument(event.document)) {
                // We need to order this after language server updates, but there's no API for that.
                // Hence, good old sleep().
                void sleep(10).then(() => this.eventEmitter.fire(this.uri));
            }
        }

        private onDidChangeActiveTextEditor(editor: vscode.TextEditor | undefined) {
            if (editor && isRustEditor(editor)) {
                this.eventEmitter.fire(this.uri);
            }
        }

        async provideTextDocumentContent(
            _uri: vscode.Uri,
            ct: vscode.CancellationToken,
        ): Promise<string> {
            const rustEditor = ctx.activeRustEditor;
            if (!rustEditor) return "";
            const client = ctx.client;

            const params = {
                textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(
                    rustEditor.document,
                ),
            };
            return client.sendRequest(ra.viewItemTree, params, ct);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-item-tree", tdcp),
    );

    return async () => {
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        tdcp.eventEmitter.fire(tdcp.uri);
        void (await vscode.window.showTextDocument(document, {
            viewColumn: vscode.ViewColumn.Two,
            preserveFocus: true,
        }));
    };
}

function crateGraph(ctx: CtxInit, full: boolean): Cmd {
    return async () => {
        const nodeModulesPath = vscode.Uri.file(path.join(ctx.extensionPath, "node_modules"));

        const panel = vscode.window.createWebviewPanel(
            "rust-analyzer.crate-graph",
            "rust-analyzer crate graph",
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [nodeModulesPath],
            },
        );
        const params = {
            full: full,
        };
        const client = ctx.client;
        const dot = await client.sendRequest(ra.viewCrateGraph, params);
        const uri = panel.webview.asWebviewUri(nodeModulesPath);

        const html = `
            <!DOCTYPE html>
            <meta charset="utf-8">
            <head>
                <style>
                    /* Fill the entire view */
                    html, body { margin:0; padding:0; overflow:hidden }
                    svg { position:fixed; top:0; left:0; height:100%; width:100% }

                    /* Disable the graphviz background and fill the polygons */
                    .graph > polygon { display:none; }
                    :is(.node,.edge) polygon { fill: white; }

                    /* Invert the line colours for dark themes */
                    body:not(.vscode-light) .edge path { stroke: white; }
                </style>
            </head>
            <body>
                <script type="text/javascript" src="${uri}/d3/dist/d3.min.js"></script>
                <script type="text/javascript" src="${uri}/@hpcc-js/wasm/dist/graphviz.umd.js"></script>
                <script type="text/javascript" src="${uri}/d3-graphviz/build/d3-graphviz.min.js"></script>
                <div id="graph"></div>
                <script>
                    let dot = \`${dot}\`;
                    let graph = d3.select("#graph")
                                  .graphviz({ useWorker: false, useSharedWorker: false })
                                  .fit(true)
                                  .zoomScaleExtent([0.1, Infinity])
                                  .renderDot(dot);

                    d3.select(window).on("click", (event) => {
                        if (event.ctrlKey) {
                            graph.resetZoom(d3.transition().duration(100));
                        }
                    });
                    d3.select(window).on("copy", (event) => {
                        event.clipboardData.setData("text/plain", dot);
                        event.preventDefault();
                    });
                </script>
            </body>
            `;

        panel.webview.html = html;
    };
}

export function viewCrateGraph(ctx: CtxInit): Cmd {
    return crateGraph(ctx, false);
}

export function viewFullCrateGraph(ctx: CtxInit): Cmd {
    return crateGraph(ctx, true);
}

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export function expandMacro(ctx: CtxInit): Cmd {
    function codeFormat(expanded: ra.ExpandedMacro): string {
        let result = `// Recursive expansion of ${expanded.name} macro\n`;
        result += "// " + "=".repeat(result.length - 3);
        result += "\n\n";
        result += expanded.expansion;

        return result;
    }

    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        uri = vscode.Uri.parse("rust-analyzer-expand-macro://expandMacro/[EXPANSION].rs");
        eventEmitter = new vscode.EventEmitter<vscode.Uri>();
        async provideTextDocumentContent(_uri: vscode.Uri): Promise<string> {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return "";
            const client = ctx.client;

            const position = editor.selection.active;

            const expanded = await client.sendRequest(ra.expandMacro, {
                textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(
                    editor.document,
                ),
                position,
            });

            if (expanded == null) return "Not available";

            return codeFormat(expanded);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-expand-macro", tdcp),
    );

    return async () => {
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        tdcp.eventEmitter.fire(tdcp.uri);
        return vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true);
    };
}

export function reloadWorkspace(ctx: CtxInit): Cmd {
    return async () => ctx.client.sendRequest(ra.reloadWorkspace);
}

export function rebuildProcMacros(ctx: CtxInit): Cmd {
    return async () => ctx.client.sendRequest(ra.rebuildProcMacros);
}

async function showReferencesImpl(
    client: LanguageClient | undefined,
    uri: string,
    position: lc.Position,
    locations: lc.Location[],
) {
    if (client) {
        await vscode.commands.executeCommand(
            "editor.action.showReferences",
            vscode.Uri.parse(uri),
            client.protocol2CodeConverter.asPosition(position),
            locations.map(client.protocol2CodeConverter.asLocation),
        );
    }
}

export function showReferences(ctx: CtxInit): Cmd {
    return async (uri: string, position: lc.Position, locations: lc.Location[]) => {
        await showReferencesImpl(ctx.client, uri, position, locations);
    };
}

export function applyActionGroup(_ctx: CtxInit): Cmd {
    return async (actions: { label: string; arguments: lc.CodeAction }[]) => {
        const selectedAction = await vscode.window.showQuickPick(actions);
        if (!selectedAction) return;
        await vscode.commands.executeCommand(
            "rust-analyzer.resolveCodeAction",
            selectedAction.arguments,
        );
    };
}

export function gotoLocation(ctx: CtxInit): Cmd {
    return async (locationLink: lc.LocationLink) => {
        const client = ctx.client;
        const uri = client.protocol2CodeConverter.asUri(locationLink.targetUri);
        let range = client.protocol2CodeConverter.asRange(locationLink.targetSelectionRange);
        // collapse the range to a cursor position
        range = range.with({ end: range.start });

        await vscode.window.showTextDocument(uri, { selection: range });
    };
}

export function openDocs(ctx: CtxInit): Cmd {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const client = ctx.client;

        const position = editor.selection.active;
        const textDocument = { uri: editor.document.uri.toString() };

        const docLinks = await client.sendRequest(ra.openDocs, { position, textDocument });
        log.debug(docLinks);

        let fileType = vscode.FileType.Unknown;
        if (docLinks.local !== undefined) {
            try {
                fileType = (await vscode.workspace.fs.stat(vscode.Uri.parse(docLinks.local))).type;
            } catch (e) {
                log.debug("stat() threw error. Falling back to web version", e);
            }
        }

        let docLink = fileType & vscode.FileType.File ? docLinks.local : docLinks.web;
        if (docLink) {
            // instruct vscode to handle the vscode-remote link directly
            if (docLink.startsWith("vscode-remote://")) {
                docLink = docLink.replace("vscode-remote://", "vscode://vscode-remote/");
            }
            const docUri = vscode.Uri.parse(docLink);
            await vscode.env.openExternal(docUri);
        }
    };
}

export function openExternalDocs(ctx: CtxInit): Cmd {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }
        const client = ctx.client;

        const position = editor.selection.active;
        const textDocument = { uri: editor.document.uri.toString() };

        const docLinks = await client.sendRequest(ra.openDocs, { position, textDocument });

        let docLink = docLinks.web;
        if (docLink) {
            // instruct vscode to handle the vscode-remote link directly
            if (docLink.startsWith("vscode-remote://")) {
                docLink = docLink.replace("vscode-remote://", "vscode://vscode-remote/");
            }
            const docUri = vscode.Uri.parse(docLink);
            await vscode.env.openExternal(docUri);
        }
    };
}

export function cancelFlycheck(ctx: CtxInit): Cmd {
    return async () => {
        await ctx.client.sendNotification(ra.cancelFlycheck);
    };
}

export function clearFlycheck(ctx: CtxInit): Cmd {
    return async () => {
        await ctx.client.sendNotification(ra.clearFlycheck);
    };
}

export function runFlycheck(ctx: CtxInit): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        const client = ctx.client;
        const params = editor ? { uri: editor.document.uri.toString() } : null;

        await client.sendNotification(ra.runFlycheck, { textDocument: params });
    };
}

export function resolveCodeAction(ctx: CtxInit): Cmd {
    return async (params: lc.CodeAction) => {
        const client = ctx.client;
        params.command = undefined;
        const item = await client.sendRequest(lc.CodeActionResolveRequest.type, params);
        if (!item?.edit) {
            return;
        }
        const itemEdit = item.edit;
        // filter out all text edits and recreate the WorkspaceEdit without them so we can apply
        // snippet edits on our own
        const lcFileSystemEdit = {
            ...itemEdit,
            documentChanges: itemEdit.documentChanges?.filter((change) => "kind" in change),
        };
        const fileSystemEdit =
            await client.protocol2CodeConverter.asWorkspaceEdit(lcFileSystemEdit);
        await vscode.workspace.applyEdit(fileSystemEdit);

        // replace all text edits so that we can convert snippet text edits into `vscode.SnippetTextEdit`s
        // FIXME: this is a workaround until vscode-languageclient supports doing the SnippeTextEdit conversion itself
        // also need to carry the snippetTextDocumentEdits separately, since we can't retrieve them again using WorkspaceEdit.entries
        const [workspaceTextEdit, snippetTextDocumentEdits] = asWorkspaceSnippetEdit(ctx, itemEdit);
        await applySnippetWorkspaceEdit(workspaceTextEdit, snippetTextDocumentEdits);
        if (item.command != null) {
            await vscode.commands.executeCommand(item.command.command, item.command.arguments);
        }
    };
}

function asWorkspaceSnippetEdit(
    ctx: CtxInit,
    item: lc.WorkspaceEdit,
): [vscode.WorkspaceEdit, SnippetTextDocumentEdit[]] {
    const client = ctx.client;

    // partially borrowed from https://github.com/microsoft/vscode-languageserver-node/blob/295aaa393fda8ecce110c38880a00466b9320e63/client/src/common/protocolConverter.ts#L1060-L1101
    const result = new vscode.WorkspaceEdit();

    if (item.documentChanges) {
        const snippetTextDocumentEdits: SnippetTextDocumentEdit[] = [];

        for (const change of item.documentChanges) {
            if (lc.TextDocumentEdit.is(change)) {
                const uri = client.protocol2CodeConverter.asUri(change.textDocument.uri);
                const snippetTextEdits: (vscode.TextEdit | vscode.SnippetTextEdit)[] = [];

                for (const edit of change.edits) {
                    if (
                        "insertTextFormat" in edit &&
                        edit.insertTextFormat === lc.InsertTextFormat.Snippet
                    ) {
                        // is a snippet text edit
                        snippetTextEdits.push(
                            new vscode.SnippetTextEdit(
                                client.protocol2CodeConverter.asRange(edit.range),
                                new vscode.SnippetString(edit.newText),
                            ),
                        );
                    } else {
                        // always as a text document edit
                        snippetTextEdits.push(
                            vscode.TextEdit.replace(
                                client.protocol2CodeConverter.asRange(edit.range),
                                edit.newText,
                            ),
                        );
                    }
                }

                snippetTextDocumentEdits.push([uri, snippetTextEdits]);
            }
        }
        return [result, snippetTextDocumentEdits];
    } else {
        // we don't handle WorkspaceEdit.changes since it's not relevant for code actions
        return [result, []];
    }
}

export function applySnippetWorkspaceEditCommand(_ctx: CtxInit): Cmd {
    return async (edit: vscode.WorkspaceEdit) => {
        await applySnippetWorkspaceEdit(edit, edit.entries());
    };
}

export function run(ctx: CtxInit, mode?: "cursor"): Cmd {
    let prevRunnable: RunnableQuickPick | undefined;

    return async () => {
        const item = await selectRunnable(ctx, prevRunnable, false, true, mode);
        if (!item) return;

        item.detail = "rerun";
        prevRunnable = item;
        const task = await createTaskFromRunnable(item.runnable, ctx.config);
        return await vscode.tasks.executeTask(task);
    };
}

export function peekTests(ctx: CtxInit): Cmd {
    return async () => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;
        const client = ctx.client;

        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: "Looking for tests...",
                cancellable: false,
            },
            async (_progress, _token) => {
                const uri = editor.document.uri.toString();
                const position = client.code2ProtocolConverter.asPosition(editor.selection.active);

                const tests = await client.sendRequest(ra.relatedTests, {
                    textDocument: { uri: uri },
                    position: position,
                });
                const locations: lc.Location[] = tests.map((it) =>
                    lc.Location.create(
                        it.runnable.location!.targetUri,
                        it.runnable.location!.targetSelectionRange,
                    ),
                );

                await showReferencesImpl(client, uri, position, locations);
            },
        );
    };
}

function isUpdatingTest(runnable: ra.Runnable): boolean {
    if (!isCargoRunnableArgs(runnable.args)) {
        return false;
    }

    const env = runnable.args.environment;
    return env ? ["UPDATE_EXPECT", "INSTA_UPDATE", "SNAPSHOTS"].some((key) => key in env) : false;
}

export function runSingle(ctx: CtxInit): Cmd {
    return async (runnable: ra.Runnable) => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        if (isUpdatingTest(runnable) && ctx.config.askBeforeUpdateTest) {
            const selection = await vscode.window.showInformationMessage(
                "rust-analyzer",
                { detail: "Do you want to update tests?", modal: true },
                "Update Now",
                "Update (and Don't ask again)",
            );

            if (selection !== "Update Now" && selection !== "Update (and Don't ask again)") {
                return;
            }

            if (selection === "Update (and Don't ask again)") {
                await ctx.config.setAskBeforeUpdateTest(false);
            }
        }

        const task = await createTaskFromRunnable(runnable, ctx.config);
        task.group = vscode.TaskGroup.Build;
        task.presentationOptions = {
            reveal: vscode.TaskRevealKind.Always,
            panel: vscode.TaskPanelKind.Dedicated,
            clear: true,
        };

        return vscode.tasks.executeTask(task);
    };
}

export function copyRunCommandLine(ctx: CtxInit) {
    let prevRunnable: RunnableQuickPick | undefined;
    return async () => {
        const item = await selectRunnable(ctx, prevRunnable);
        if (!item || !isCargoRunnableArgs(item.runnable.args)) return;
        const args = createCargoArgs(item.runnable.args);
        const commandLine = ["cargo", ...args].join(" ");
        await vscode.env.clipboard.writeText(commandLine);
        await vscode.window.showInformationMessage("Cargo invocation copied to the clipboard.");
    };
}

export function debug(ctx: CtxInit): Cmd {
    let prevDebuggee: RunnableQuickPick | undefined;

    return async () => {
        const item = await selectRunnable(ctx, prevDebuggee, true);
        if (!item) return;

        item.detail = "restart";
        prevDebuggee = item;
        return await startDebugSession(ctx, item.runnable);
    };
}

export function debugSingle(ctx: CtxInit): Cmd {
    return async (config: ra.Runnable) => {
        await startDebugSession(ctx, config);
    };
}

export function newDebugConfig(ctx: CtxInit): Cmd {
    return async () => {
        const item = await selectRunnable(ctx, undefined, true, false);
        if (!item) return;

        await makeDebugConfig(ctx, item.runnable);
    };
}

export function hoverRefCommandProxy(_: Ctx): Cmd {
    return async (index: number) => {
        const link = HOVER_REFERENCE_COMMAND[index];
        if (link) {
            const { command, arguments: args = [] } = link;
            await vscode.commands.executeCommand(command, ...args);
        }
    };
}

export function viewMemoryLayout(ctx: CtxInit): Cmd {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        const client = ctx.client;

        const position = editor.selection.active;
        const expanded = await client.sendRequest(ra.viewRecursiveMemoryLayout, {
            textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(editor.document),
            position,
        });

        const document = vscode.window.createWebviewPanel(
            "memory_layout",
            "[Memory Layout]",
            vscode.ViewColumn.Two,
            { enableScripts: true },
        );

        document.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            overflow: hidden;
            min-height: 100%;
            height: 100vh;
            padding: 32px;
            position: relative;
            display: block;

            background-color: var(--vscode-editor-background);
            font-family: var(--vscode-editor-font-family);
            font-size: var(--vscode-editor-font-size);
            color: var(--vscode-editor-foreground);
        }

        .container {
            position: relative;
        }

        .trans {
            transition: all 0.2s ease-in-out;
        }

        .grid {
            height: 100%;
            position: relative;
            color: var(--vscode-commandCenter-activeBorder);
            pointer-events: none;
        }

        .grid-line {
            position: absolute;
            width: 100%;
            height: 1px;
            background-color: var(--vscode-commandCenter-activeBorder);
        }

        #tooltip {
            position: fixed;
            display: none;
            z-index: 1;
            pointer-events: none;
            padding: 4px 8px;
            z-index: 2;

            color: var(--vscode-editorHoverWidget-foreground);
            background-color: var(--vscode-editorHoverWidget-background);
            border: 1px solid var(--vscode-editorHoverWidget-border);
        }

        #tooltip b {
            color: var(--vscode-editorInlayHint-typeForeground);
        }

        #tooltip ul {
            margin-left: 0;
            padding-left: 20px;
        }

        table {
            position: absolute;
            transform: rotateZ(90deg) rotateX(180deg);
            transform-origin: top left;
            border-collapse: collapse;
            table-layout: fixed;
            left: 48px;
            top: 0;
            max-height: calc(100vw - 64px - 48px);
            z-index: 1;
        }

        td {
            border: 1px solid var(--vscode-focusBorder);
            writing-mode: vertical-rl;
            text-orientation: sideways-right;

            height: 80px;
        }

        td p {
            height: calc(100% - 16px);
            width: calc(100% - 8px);
            margin: 8px 4px;
            display: inline-block;
            transform: rotateY(180deg);
            pointer-events: none;
            overflow: hidden;
        }

        td p * {
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            display: inline-block;
            height: 100%;
        }

        td p b {
            color: var(--vscode-editorInlayHint-typeForeground);
        }

        td:hover {
            background-color: var(--vscode-editor-hoverHighlightBackground);
        }

        td:empty {
            visibility: hidden;
            border: 0;
        }
    </style>
</head>
<body>
    <div id="tooltip"></div>
</body>
<script>(function() {

const data = ${JSON.stringify(expanded)}

if (!(data && data.nodes.length)) {
    document.body.innerText = "Not Available"
    return
}

data.nodes.map(n => {
    n.typename = n.typename.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', ' & quot; ').replaceAll("'", '&#039;')
    return n
})

let height = window.innerHeight - 64

addEventListener("resize", e => {
    const new_height = window.innerHeight - 64
    height = new_height
    container.classList.remove("trans")
    table.classList.remove("trans")
    locate()
    setTimeout(() => { // give delay to redraw, annoying but needed
        container.classList.add("trans")
        table.classList.add("trans")
    }, 0)
})

const container = document.createElement("div")
container.classList.add("container")
container.classList.add("trans")
document.body.appendChild(container)

const tooltip = document.getElementById("tooltip")

let y = 0
let zoom = 1.0

const table = document.createElement("table")
table.classList.add("trans")
container.appendChild(table)
const rows = []

function node_t(idx, depth, offset) {
    if (!rows[depth]) {
        rows[depth] = { el: document.createElement("tr"), offset: 0 }
    }

    if (rows[depth].offset < offset) {
        const pad = document.createElement("td")
        pad.colSpan = offset - rows[depth].offset
        rows[depth].el.appendChild(pad)
        rows[depth].offset += offset - rows[depth].offset
    }

    const td = document.createElement("td")
    td.innerHTML = '<p><span>' + data.nodes[idx].itemName + ':</span> <b>' + data.nodes[idx].typename + '</b></p>'

    td.colSpan = data.nodes[idx].size

    td.addEventListener("mouseover", e => {
        const node = data.nodes[idx]
        tooltip.innerHTML = node.itemName + ": <b>" + node.typename + "</b><br/>"
            + "<ul>"
            + "<li>size = " + node.size + "</li>"
            + "<li>align = " + node.alignment + "</li>"
            + "<li>field offset = " + node.offset + "</li>"
            + "</ul>"
            + "<i>double click to focus</i>"

        tooltip.style.display = "block"
    })
    td.addEventListener("mouseleave", _ => tooltip.style.display = "none")
    const total_offset = rows[depth].offset
    td.addEventListener("dblclick", e => {
        const node = data.nodes[idx]
        zoom = data.nodes[0].size / node.size
        y = -(total_offset) / data.nodes[0].size * zoom
        x = 0
        locate()
    })

    rows[depth].el.appendChild(td)
    rows[depth].offset += data.nodes[idx].size


    if (data.nodes[idx].childrenStart != -1) {
        for (let i = 0; i < data.nodes[idx].childrenLen; i++) {
            if (data.nodes[data.nodes[idx].childrenStart + i].size) {
                node_t(data.nodes[idx].childrenStart + i, depth + 1, offset + data.nodes[data.nodes[idx].childrenStart + i].offset)
            }
        }
    }
}

node_t(0, 0, 0)

for (const row of rows) table.appendChild(row.el)

const grid = document.createElement("div")
grid.classList.add("grid")
container.appendChild(grid)

for (let i = 0; i < data.nodes[0].size / 8 + 1; i++) {
    const el = document.createElement("div")
    el.classList.add("grid-line")
    el.style.top = (i / (data.nodes[0].size / 8) * 100) + "%"
    el.innerText = i * 8
    grid.appendChild(el)
}

addEventListener("mousemove", e => {
    tooltip.style.top = e.clientY + 10 + "px"
    tooltip.style.left = e.clientX + 10 + "px"
})

function locate() {
    container.style.top = height * y + "px"
    container.style.height = (height * zoom) + "px"

    table.style.width = container.style.height
}

locate()

})()
</script>
</html>`;

        ctx.pushExtCleanup(document);
    };
}

export function toggleCheckOnSave(ctx: Ctx): Cmd {
    return async () => {
        await ctx.config.toggleCheckOnSave();
        ctx.refreshServerStatus();
    };
}

export function toggleLSPLogs(ctx: Ctx): Cmd {
    return async () => {
        const config = vscode.workspace.getConfiguration("rust-analyzer");
        const targetValue =
            config.get<string | undefined>("trace.server") === "verbose" ? undefined : "verbose";

        await config.update("trace.server", targetValue, vscode.ConfigurationTarget.Workspace);
        if (targetValue && ctx.client && ctx.client.traceOutputChannel) {
            ctx.client.traceOutputChannel.show();
        }
    };
}

export function openWalkthrough(_: Ctx): Cmd {
    return async () => {
        await vscode.commands.executeCommand(
            "workbench.action.openWalkthrough",
            "rust-lang.rust-analyzer#landing",
            false,
        );
    };
}
