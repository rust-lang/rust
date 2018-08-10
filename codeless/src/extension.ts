'use strict';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    Executable,
    TextDocumentIdentifier,
    Range
} from 'vscode-languageclient';


let client: LanguageClient;

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
                let r: Range = { start: s.start, end: s.end }
                return r;
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
    let run: Executable = {
        command: "cargo",
        args: ["run", "--package", "m"],
        options: { cwd: "." }
    }
    let serverOptions: ServerOptions = {
        run,
        debug: run
    };

    let clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'rust' }],
    };

    client = new LanguageClient(
        'm',
        'm languge server',
        serverOptions,
        clientOptions,
    );
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

interface SyntaxTreeParams {
    textDocument: TextDocumentIdentifier;
}

type SyntaxTreeResult = string

interface ExtendSelectionParams {
    textDocument: TextDocumentIdentifier;
    selections: Range[];
}

interface ExtendSelectionResult {
    selections: Range[];
}
