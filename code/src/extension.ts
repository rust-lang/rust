'use strict';
import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient'


let client: lc.LanguageClient;

let uris = {
    syntaxTree: vscode.Uri.parse('libsyntax-rust://syntaxtree')
}


export function activate(context: vscode.ExtensionContext) {
    let textDocumentContentProvider = new TextDocumentContentProvider()
    let dispose = (disposable) => {
        context.subscriptions.push(disposable);
    }
    let registerCommand = (name, f) => {
        dispose(vscode.commands.registerCommand(name, f))
    }

    registerCommand('libsyntax-rust.syntaxTree', () => openDoc(uris.syntaxTree))
    registerCommand('libsyntax-rust.extendSelection', async () => {
        let editor = vscode.window.activeTextEditor
        if (editor == null || editor.document.languageId != "rust") return
        let request: ExtendSelectionParams = {
            textDocument: { uri: editor.document.uri.toString() },
            selections: editor.selections.map((s) => {
                return { start: s.start, end: s.end };
            })
        }
        let response = await client.sendRequest<ExtendSelectionResult>("m/extendSelection", request)
        editor.selections = response.selections.map((range) => {
            return new vscode.Selection(
                new vscode.Position(range.start.line, range.start.character),
                new vscode.Position(range.end.line, range.end.character),
            )
        })
    })

    dispose(vscode.workspace.registerTextDocumentContentProvider(
        'libsyntax-rust',
        textDocumentContentProvider
    ))
    startServer()
    vscode.workspace.onDidChangeTextDocument((event: vscode.TextDocumentChangeEvent) => {
        let doc = event.document
        if (doc.languageId != "rust") return
        // We need to order this after LS updates, but there's no API for that.
        // Hence, good old setTimeout.
        setTimeout(() => {
            textDocumentContentProvider.eventEmitter.fire(uris.syntaxTree)
        }, 10)
    }, null, context.subscriptions)
}

export function deactivate(): Thenable<void> {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

function startServer() {
    let run: lc.Executable = {
        command: "m",
        // args: ["run", "--package", "m"],
        options: { cwd: "." }
    }
    let serverOptions: lc.ServerOptions = {
        run,
        debug: run
    };

    let clientOptions: lc.LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'rust' }],
    };

    client = new lc.LanguageClient(
        'm',
        'm languge server',
        serverOptions,
        clientOptions,
    );
    client.onReady().then(() => {
        client.onNotification(
            new lc.NotificationType("m/publishDecorations"),
            (params: PublishDecorationsParams) => {
                let editor = vscode.window.visibleTextEditors.find(
                    (editor) => editor.document.uri.toString() == params.uri
                )
                if (editor == null) return;
                setHighlights(
                    editor,
                    params.decorations,
                )
            }
        )
    })
    client.start();
}

async function openDoc(uri: vscode.Uri) {
    let document = await vscode.workspace.openTextDocument(uri)
    return vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true)
}

class TextDocumentContentProvider implements vscode.TextDocumentContentProvider {
    public eventEmitter = new vscode.EventEmitter<vscode.Uri>()
    public syntaxTree: string = "Not available"

    public provideTextDocumentContent(uri: vscode.Uri): vscode.ProviderResult<string> {
        let editor = vscode.window.activeTextEditor;
        if (editor == null) return ""
        let request: SyntaxTreeParams = {
            textDocument: { uri: editor.document.uri.toString() }
        };
        return client.sendRequest<SyntaxTreeResult>("m/syntaxTree", request);
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event
    }
}


const decorations = (() => {
    const decor = (obj) => vscode.window.createTextEditorDecorationType({ color: obj })
    return {
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
})()

function setHighlights(
    editor: vscode.TextEditor,
    highlihgs: Array<Decoration>
) {
    let byTag = {}
    for (let tag in decorations) {
        byTag[tag] = []
    }

    for (let d of highlihgs) {
        if (!byTag[d.tag]) {
            console.log(`unknown tag ${d.tag}`)
            continue
        }
        byTag[d.tag].push(d.range)
    }

    for (let tag in byTag) {
        let dec = decorations[tag]
        let ranges = byTag[tag]
        editor.setDecorations(dec, ranges)
    }
}

interface SyntaxTreeParams {
    textDocument: lc.TextDocumentIdentifier;
}

type SyntaxTreeResult = string

interface ExtendSelectionParams {
    textDocument: lc.TextDocumentIdentifier;
    selections: lc.Range[];
}

interface ExtendSelectionResult {
    selections: lc.Range[];
}

interface PublishDecorationsParams {
    uri: string,
    decorations: Decoration[],
}

interface Decoration {
    range: lc.Range,
    tag: string,
}
