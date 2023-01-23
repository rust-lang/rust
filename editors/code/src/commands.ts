import * as vscode from "vscode";
import * as lc from "vscode-languageclient";
import * as ra from "./lsp_ext";
import * as path from "path";

import { Ctx, Cmd, CtxInit } from "./ctx";
import { applySnippetWorkspaceEdit, applySnippetTextEdits } from "./snippets";
import { spawnSync } from "child_process";
import { RunnableQuickPick, selectRunnable, createTask, createArgs } from "./run";
import { AstInspector } from "./ast_inspector";
import { isRustDocument, isCargoTomlDocument, sleep, isRustEditor } from "./util";
import { startDebugSession, makeDebugConfig } from "./debug";
import { LanguageClient } from "vscode-languageclient/node";
import { LINKED_COMMANDS } from "./client";

export * from "./ast_inspector";
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
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-status", tdcp)
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

            return ctx.client.sendRequest(ra.memoryUsage).then((mem: any) => {
                return "Per-query memory usage:\n" + mem + "\n(note: database has been cleared)";
            });
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-memory", tdcp)
    );

    return async () => {
        tdcp.eventEmitter.fire(tdcp.uri);
        const document = await vscode.workspace.openTextDocument(tdcp.uri);
        return vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true);
    };
}

export function shuffleCrateGraph(ctx: CtxInit): Cmd {
    return async () => {
        return ctx.client.sendRequest(ra.shuffleCrateGraph);
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
                client.code2ProtocolConverter.asPosition(s.active)
            ),
        });
        editor.selections = editor.selections.map((sel, idx) => {
            const active = client.protocol2CodeConverter.asPosition(response[idx]);
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
            textEdits.forEach((edit: any) => {
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
                    editor.document
                ),
                position: client.code2ProtocolConverter.asPosition(editor.selection.active),
            })
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
            const loc = locations[0];

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
                locations.map((loc) => lc.Location.create(loc.targetUri, loc.targetRange))
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

export function ssr(ctx: CtxInit): Cmd {
    return async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const client = ctx.client;

        const position = editor.selection.active;
        const selections = editor.selections;
        const textDocument = client.code2ProtocolConverter.asTextDocumentIdentifier(
            editor.document
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
                    return e.toString();
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
                    await client.protocol2CodeConverter.asWorkspaceEdit(edit, token)
                );
            }
        );
    };
}

export function serverVersion(ctx: CtxInit): Cmd {
    return async () => {
        if (!ctx.serverPath) {
            void vscode.window.showWarningMessage(`rust-analyzer server is not running`);
            return;
        }
        const { stdout } = spawnSync(ctx.serverPath, ["--version"], { encoding: "utf8" });
        const versionString = stdout.slice(`rust-analyzer `.length).trim();

        void vscode.window.showInformationMessage(`rust-analyzer version: ${versionString}`);
    };
}

// Opens the virtual file that will show the syntax tree
//
// The contents of the file come from the `TextDocumentContentProvider`
export function syntaxTree(ctx: CtxInit): Cmd {
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse("rust-analyzer-syntax-tree://syntaxtree/tree.rast");
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();
        constructor() {
            vscode.workspace.onDidChangeTextDocument(
                this.onDidChangeTextDocument,
                this,
                ctx.subscriptions
            );
            vscode.window.onDidChangeActiveTextEditor(
                this.onDidChangeActiveTextEditor,
                this,
                ctx.subscriptions
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
            uri: vscode.Uri,
            ct: vscode.CancellationToken
        ): Promise<string> {
            const rustEditor = ctx.activeRustEditor;
            if (!rustEditor) return "";
            const client = ctx.client;

            // When the range based query is enabled we take the range of the selection
            const range =
                uri.query === "range=true" && !rustEditor.selection.isEmpty
                    ? client.code2ProtocolConverter.asRange(rustEditor.selection)
                    : null;

            const params = { textDocument: { uri: rustEditor.document.uri.toString() }, range };
            return client.sendRequest(ra.syntaxTree, params, ct);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(new AstInspector(ctx));
    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-syntax-tree", tdcp)
    );
    ctx.pushExtCleanup(
        vscode.languages.setLanguageConfiguration("ra_syntax_tree", {
            brackets: [["[", ")"]],
        })
    );

    return async () => {
        const editor = vscode.window.activeTextEditor;
        const rangeEnabled = !!editor && !editor.selection.isEmpty;

        const uri = rangeEnabled ? vscode.Uri.parse(`${tdcp.uri.toString()}?range=true`) : tdcp.uri;

        const document = await vscode.workspace.openTextDocument(uri);

        tdcp.eventEmitter.fire(uri);

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
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse("rust-analyzer-hir://viewHir/hir.rs");
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();
        constructor() {
            vscode.workspace.onDidChangeTextDocument(
                this.onDidChangeTextDocument,
                this,
                ctx.subscriptions
            );
            vscode.window.onDidChangeActiveTextEditor(
                this.onDidChangeActiveTextEditor,
                this,
                ctx.subscriptions
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
            ct: vscode.CancellationToken
        ): Promise<string> {
            const rustEditor = ctx.activeRustEditor;
            if (!rustEditor) return "";

            const client = ctx.client;
            const params = {
                textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(
                    rustEditor.document
                ),
                position: client.code2ProtocolConverter.asPosition(rustEditor.selection.active),
            };
            return client.sendRequest(ra.viewHir, params, ct);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-hir", tdcp)
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

export function viewFileText(ctx: CtxInit): Cmd {
    const tdcp = new (class implements vscode.TextDocumentContentProvider {
        readonly uri = vscode.Uri.parse("rust-analyzer-file-text://viewFileText/file.rs");
        readonly eventEmitter = new vscode.EventEmitter<vscode.Uri>();
        constructor() {
            vscode.workspace.onDidChangeTextDocument(
                this.onDidChangeTextDocument,
                this,
                ctx.subscriptions
            );
            vscode.window.onDidChangeActiveTextEditor(
                this.onDidChangeActiveTextEditor,
                this,
                ctx.subscriptions
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
            ct: vscode.CancellationToken
        ): Promise<string> {
            const rustEditor = ctx.activeRustEditor;
            if (!rustEditor) return "";
            const client = ctx.client;

            const params = client.code2ProtocolConverter.asTextDocumentIdentifier(
                rustEditor.document
            );
            return client.sendRequest(ra.viewFileText, params, ct);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-file-text", tdcp)
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
                ctx.subscriptions
            );
            vscode.window.onDidChangeActiveTextEditor(
                this.onDidChangeActiveTextEditor,
                this,
                ctx.subscriptions
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
            ct: vscode.CancellationToken
        ): Promise<string> {
            const rustEditor = ctx.activeRustEditor;
            if (!rustEditor) return "";
            const client = ctx.client;

            const params = {
                textDocument: client.code2ProtocolConverter.asTextDocumentIdentifier(
                    rustEditor.document
                ),
            };
            return client.sendRequest(ra.viewItemTree, params, ct);
        }

        get onDidChange(): vscode.Event<vscode.Uri> {
            return this.eventEmitter.event;
        }
    })();

    ctx.pushExtCleanup(
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-item-tree", tdcp)
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
            }
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
                <script type="text/javascript" src="${uri}/@hpcc-js/wasm/dist/index.min.js"></script>
                <script type="text/javascript" src="${uri}/d3-graphviz/build/d3-graphviz.min.js"></script>
                <div id="graph"></div>
                <script>
                    let graph = d3.select("#graph")
                                  .graphviz()
                                  .fit(true)
                                  .zoomScaleExtent([0.1, Infinity])
                                  .renderDot(\`${dot}\`);

                    d3.select(window).on("click", (event) => {
                        if (event.ctrlKey) {
                            graph.resetZoom(d3.transition().duration(100));
                        }
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
        let result = `// Recursive expansion of ${expanded.name}! macro\n`;
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
                    editor.document
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
        vscode.workspace.registerTextDocumentContentProvider("rust-analyzer-expand-macro", tdcp)
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

async function showReferencesImpl(
    client: LanguageClient | undefined,
    uri: string,
    position: lc.Position,
    locations: lc.Location[]
) {
    if (client) {
        await vscode.commands.executeCommand(
            "editor.action.showReferences",
            vscode.Uri.parse(uri),
            client.protocol2CodeConverter.asPosition(position),
            locations.map(client.protocol2CodeConverter.asLocation)
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
            selectedAction.arguments
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

        const doclink = await client.sendRequest(ra.openDocs, { position, textDocument });

        if (doclink != null) {
            await vscode.commands.executeCommand("vscode.open", vscode.Uri.parse(doclink));
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
        const edit = await client.protocol2CodeConverter.asWorkspaceEdit(itemEdit);
        // filter out all text edits and recreate the WorkspaceEdit without them so we can apply
        // snippet edits on our own
        const lcFileSystemEdit = {
            ...itemEdit,
            documentChanges: itemEdit.documentChanges?.filter((change) => "kind" in change),
        };
        const fileSystemEdit = await client.protocol2CodeConverter.asWorkspaceEdit(
            lcFileSystemEdit
        );
        await vscode.workspace.applyEdit(fileSystemEdit);
        await applySnippetWorkspaceEdit(edit);
        if (item.command != null) {
            await vscode.commands.executeCommand(item.command.command, item.command.arguments);
        }
    };
}

export function applySnippetWorkspaceEditCommand(_ctx: CtxInit): Cmd {
    return async (edit: vscode.WorkspaceEdit) => {
        await applySnippetWorkspaceEdit(edit);
    };
}

export function run(ctx: CtxInit): Cmd {
    let prevRunnable: RunnableQuickPick | undefined;

    return async () => {
        const item = await selectRunnable(ctx, prevRunnable);
        if (!item) return;

        item.detail = "rerun";
        prevRunnable = item;
        const task = await createTask(item.runnable, ctx.config);
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
                        it.runnable.location!.targetSelectionRange
                    )
                );

                await showReferencesImpl(client, uri, position, locations);
            }
        );
    };
}

export function runSingle(ctx: CtxInit): Cmd {
    return async (runnable: ra.Runnable) => {
        const editor = ctx.activeRustEditor;
        if (!editor) return;

        const task = await createTask(runnable, ctx.config);
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
        if (!item) return;
        const args = createArgs(item.runnable);
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

export function linkToCommand(_: Ctx): Cmd {
    return async (commandId: string) => {
        const link = LINKED_COMMANDS.get(commandId);
        if (link) {
            const { command, arguments: args = [] } = link;
            await vscode.commands.executeCommand(command, ...args);
        }
    };
}
