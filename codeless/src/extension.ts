'use strict';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    Executable,
    TextDocumentIdentifier
} from 'vscode-languageclient';


let client: LanguageClient;

let uris = {
    syntaxTree: vscode.Uri.parse('libsyntax-rust://syntaxtree')
}


export function activate(context: vscode.ExtensionContext) {
    let dispose = (disposable) => {
        context.subscriptions.push(disposable);
    }
    let registerCommand = (name, f) => {
        dispose(vscode.commands.registerCommand(name, f))
    }

    registerCommand('libsyntax-rust.syntaxTree', () => openDoc(uris.syntaxTree))
    dispose(vscode.workspace.registerTextDocumentContentProvider(
        'libsyntax-rust',
        new TextDocumentContentProvider()
    ))
    startServer()
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
        let textDocument: TextDocumentIdentifier = { uri: editor.document.uri.toString() };
        return client.sendRequest("m/syntaxTree", { textDocument })
    }

    get onDidChange(): vscode.Event<vscode.Uri> {
        return this.eventEmitter.event
    }
}
