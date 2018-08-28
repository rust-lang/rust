'use strict';
import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient'
import { DH_UNABLE_TO_CHECK_GENERATOR } from 'constants';


let client: lc.LanguageClient;

let uris = {
    syntaxTree: vscode.Uri.parse('libsyntax-rust://syntaxtree')
}


export function activate(context: vscode.ExtensionContext) {
    let textDocumentContentProvider = new TextDocumentContentProvider()
    let dispose = (disposable: vscode.Disposable) => {
        context.subscriptions.push(disposable);
    }
    let registerCommand = (name: string, f: any) => {
        dispose(vscode.commands.registerCommand(name, f))
    }

    registerCommand('libsyntax-rust.syntaxTree', () => openDoc(uris.syntaxTree))
    registerCommand('libsyntax-rust.extendSelection', async () => {
        let editor = vscode.window.activeTextEditor
        if (editor == null || editor.document.languageId != "rust") return
        let request: ExtendSelectionParams = {
            textDocument: { uri: editor.document.uri.toString() },
            selections: editor.selections.map((s) => {
                return client.code2ProtocolConverter.asRange(s)
            })
        }
        let response = await client.sendRequest<ExtendSelectionResult>("m/extendSelection", request)
        editor.selections = response.selections.map((range) => {
            let r = client.protocol2CodeConverter.asRange(range)
            return new vscode.Selection(r.start, r.end)
        })
    })
    registerCommand('libsyntax-rust.matchingBrace', async () => {
        let editor = vscode.window.activeTextEditor
        if (editor == null || editor.document.languageId != "rust") return
        let request: FindMatchingBraceParams = {
            textDocument: { uri: editor.document.uri.toString() },
            offsets: editor.selections.map((s) => {
                return client.code2ProtocolConverter.asPosition(s.active)
            })
        }
        let response = await client.sendRequest<lc.Position[]>("m/findMatchingBrace", request)
        editor.selections = editor.selections.map((sel, idx) => {
            let active = client.protocol2CodeConverter.asPosition(response[idx])
            let anchor = sel.isEmpty ? active : sel.anchor
            return new vscode.Selection(anchor, active)
        })
        editor.revealRange(editor.selection)
    })
    registerCommand('libsyntax-rust.joinLines', async () => {
        let editor = vscode.window.activeTextEditor
        if (editor == null || editor.document.languageId != "rust") return
        let request: JoinLinesParams = {
            textDocument: { uri: editor.document.uri.toString() },
            range: client.code2ProtocolConverter.asRange(editor.selection),
        }
        let response = await client.sendRequest<lc.TextEdit[]>("m/joinLines", request)
        let edits = client.protocol2CodeConverter.asTextEdits(response)
        let wsEdit = new vscode.WorkspaceEdit()
        wsEdit.set(editor.document.uri, edits)
        return vscode.workspace.applyEdit(wsEdit)
    })
    registerCommand('libsyntax-rust.parentModule', async () => {
        let editor = vscode.window.activeTextEditor
        if (editor == null || editor.document.languageId != "rust") return
        let request: lc.TextDocumentIdentifier = {
            uri: editor.document.uri.toString()
        }
        let response = await client.sendRequest<lc.Location[]>("m/parentModule", request)
        let loc = response[0]
        if (loc == null) return
        let uri = client.protocol2CodeConverter.asUri(loc.uri)
        let range = client.protocol2CodeConverter.asRange(loc.range)

        let doc = await vscode.workspace.openTextDocument(uri)
        let e = await vscode.window.showTextDocument(doc)
        e.revealRange(range, vscode.TextEditorRevealType.InCenter)
    })

    let prevRunnable: RunnableQuickPick | undefined = undefined
    registerCommand('libsyntax-rust.run', async () => {
        let editor = vscode.window.activeTextEditor
        if (editor == null || editor.document.languageId != "rust") return
        let textDocument: lc.TextDocumentIdentifier = {
            uri: editor.document.uri.toString()
        }
        let params: RunnablesParams = {
            textDocument,
            position: client.code2ProtocolConverter.asPosition(editor.selection.active)
        }
        let runnables = await client.sendRequest<Runnable[]>('m/runnables', params)
        let items: RunnableQuickPick[] = []
        if (prevRunnable) {
            items.push(prevRunnable)
        }
        for (let r of runnables) {
            if (prevRunnable && JSON.stringify(prevRunnable.runnable) == JSON.stringify(r)) {
                continue
            }
            items.push(new RunnableQuickPick(r))
        }
        let item = await vscode.window.showQuickPick(items)
        if (item) {
            item.detail = "rerun"
            prevRunnable = item
            let task = createTask(item.runnable)
            return await vscode.tasks.executeTask(task)
        }
    })
    registerCommand('libsyntax-rust.fsEdit', async (ops: FsOp[]) => {
        let edit = new vscode.WorkspaceEdit()
        let created;
        let moved;
        for (let op of ops) {
            if (op.type == "createFile") {
                let uri = vscode.Uri.parse(op.uri!)
                edit.createFile(uri)
                created = uri
            } else if (op.type == "moveFile") {
                let src = vscode.Uri.parse(op.src!)
                let dst = vscode.Uri.parse(op.dst!)
                edit.renameFile(src, dst)
                moved = dst
            } else {
                console.error(`unknown op: ${JSON.stringify(op)}`)
            }
        }
        await vscode.workspace.applyEdit(edit)
        let doc = await vscode.workspace.openTextDocument((created || moved)!)
        await vscode.window.showTextDocument(doc)
    })

    dispose(vscode.workspace.registerTextDocumentContentProvider(
        'libsyntax-rust',
        textDocumentContentProvider
    ))
    startServer()
    vscode.workspace.onDidChangeTextDocument((event: vscode.TextDocumentChangeEvent) => {
        let doc = event.document
        if (doc.languageId != "rust") return
        afterLs(() => {
            textDocumentContentProvider.eventEmitter.fire(uris.syntaxTree)
        })
    }, null, context.subscriptions)
    vscode.window.onDidChangeActiveTextEditor(async (editor) => {
        if (!editor || editor.document.languageId != 'rust') return
        let params: lc.TextDocumentIdentifier = {
            uri: editor.document.uri.toString()
        }
        let decorations = await client.sendRequest<Decoration[]>("m/decorationsRequest", params)
        setHighlights(editor, decorations)
    })
}

// We need to order this after LS updates, but there's no API for that.
// Hence, good old setTimeout.
function afterLs(f: () => any) {
    setTimeout(f, 10)
}

export function deactivate(): Thenable<void> {
    if (!client) {
        return Promise.resolve();
    }
    return client.stop();
}

function startServer() {
    let run: lc.Executable = {
        // command: "cargo",
        // args: ["run", "--package", "m"],
        command: "m",
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
            "m/publishDecorations",
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
        client.onRequest(
            new lc.RequestType<lc.Position, void, any, any>("m/moveCursor"),
            (params: lc.Position, token: lc.CancellationToken) => {
                let editor = vscode.window.activeTextEditor;
                if (!editor) return
                if (!editor.selection.isEmpty) return
                let position = client.protocol2CodeConverter.asPosition(params)
                afterLs(() => {
                    editor!.selection = new vscode.Selection(position, position)
                })
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


const decorations: { [index: string]: vscode.TextEditorDecorationType } = (() => {
    const decor = (obj: any) => vscode.window.createTextEditorDecorationType({ color: obj })
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
    let byTag: Map<string, vscode.Range[]> = new Map()
    for (let tag in decorations) {
        byTag.set(tag, [])
    }

    for (let d of highlihgs) {
        if (!byTag.get(d.tag)) {
            console.log(`unknown tag ${d.tag}`)
            continue
        }
        byTag.get(d.tag)!.push(
            client.protocol2CodeConverter.asRange(d.range)
        )
    }

    for (let tag of byTag.keys()) {
        let dec: vscode.TextEditorDecorationType = decorations[tag]
        let ranges = byTag.get(tag)!
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

interface FindMatchingBraceParams {
    textDocument: lc.TextDocumentIdentifier;
    offsets: lc.Position[];
}

interface JoinLinesParams {
    textDocument: lc.TextDocumentIdentifier;
    range: lc.Range;
}

interface PublishDecorationsParams {
    uri: string,
    decorations: Decoration[],
}

interface RunnablesParams {
    textDocument: lc.TextDocumentIdentifier,
    position?: lc.Position,
}

interface Runnable {
    range: lc.Range;
    label: string;
    bin: string;
    args: string[];
    env: { [index: string]: string },
}

class RunnableQuickPick implements vscode.QuickPickItem {
    label: string;
    description?: string | undefined;
    detail?: string | undefined;
    picked?: boolean | undefined;

    constructor(public runnable: Runnable) {
        this.label = runnable.label
    }
}

interface Decoration {
    range: lc.Range,
    tag: string,
}


interface CargoTaskDefinition extends vscode.TaskDefinition {
    type: 'cargo';
    label: string;
    command: string;
    args: Array<string>;
    env?: { [key: string]: string };
}

function createTask(spec: Runnable): vscode.Task {
    const TASK_SOURCE = 'Rust';
    let definition: CargoTaskDefinition = {
        type: 'cargo',
        label: 'cargo',
        command: spec.bin,
        args: spec.args,
        env: spec.env
    }

    let execCmd = `${definition.command} ${definition.args.join(' ')}`;
    let execOption: vscode.ShellExecutionOptions = {
        cwd: '.',
        env: definition.env,
    };
    let exec = new vscode.ShellExecution(`clear; ${execCmd}`, execOption);

    let f = vscode.workspace.workspaceFolders![0]
    let t = new vscode.Task(definition, f, definition.label, TASK_SOURCE, exec, ['$rustc']);
    return t;
}

interface FsOp {
    type: string;
    uri?: string;
    src?: string;
    dst?: string;
}
