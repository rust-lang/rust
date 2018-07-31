'use strict'
import * as vscode from 'vscode'

const backend = require("../../native")

let docToSyntax;

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

    docToSyntax = documentToFile(context.subscriptions, () => {
        let emitter = textDocumentContentProvider.eventEmitter
        emitter.fire(uris.syntaxTree)
        let syntax = activeSyntax()
        setHighlights(vscode.window.activeTextEditor, syntax.highlight())
    })


    dispose(vscode.workspace.registerTextDocumentContentProvider(
        'libsyntax-rust',
        textDocumentContentProvider
    ))

    registerCommand('libsyntax-rust.syntaxTree', () => openDoc(uris.syntaxTree))
}

export function deactivate() { }

export class Syntax {
    imp;
    doc: vscode.TextDocument;

    constructor(imp, doc: vscode.TextDocument) {
        this.imp = imp
        this.doc = doc
    }

    syntaxTree(): string { return this.imp.syntaxTree() }
    highlight(): Array<[number, number, string]> { return this.imp.highlight() }
}


function activeDoc() {
    return vscode.window.activeTextEditor.document
}

function activeSyntax(): Syntax {
    let doc = activeDoc()
    if (doc == null) return null
    return docToSyntax(doc)
}

async function openDoc(uri: vscode.Uri) {
    let document = await vscode.workspace.openTextDocument(uri)
    return vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true)
}

function documentToFile(disposables: vscode.Disposable[], onChange) {
    let docs = {}
    function update(doc: vscode.TextDocument, file) {
        let key = doc.uri.toString()
        if (file == null) {
            delete docs[key]
        } else {
            docs[key] = file
        }
        onChange(doc)
    }
    function get(doc: vscode.TextDocument) {
        return docs[doc.uri.toString()]
    }

    function isKnownDoc(doc: vscode.TextDocument) {
        return doc.fileName.endsWith('.rs')
    }

    function createFile(text: String) {
        console.time("parsing")
        let res = new backend.RustFile(text);
        console.timeEnd("parsing")
        return res
    }

    vscode.workspace.onDidChangeTextDocument((event: vscode.TextDocumentChangeEvent) => {
        let doc = event.document
        if (!isKnownDoc(event.document)) return
        update(doc, null)
    }, null, disposables)

    vscode.workspace.onDidOpenTextDocument((doc: vscode.TextDocument) => {
        if (!isKnownDoc(doc)) return
        update(doc, createFile(doc.getText()))
    }, null, disposables)

    vscode.workspace.onDidCloseTextDocument((doc: vscode.TextDocument) => {
        update(doc, null)
    }, null, disposables)

    return (doc: vscode.TextDocument) => {
        if (!isKnownDoc(doc)) return null

        if (!get(doc)) {
            update(doc, createFile(doc.getText()))
        }
        let imp = get(doc)
        return new Syntax(imp, doc)
    }
}

export class TextDocumentContentProvider implements vscode.TextDocumentContentProvider {
    public eventEmitter = new vscode.EventEmitter<vscode.Uri>()
    public syntaxTree: string = "Not available"

    public provideTextDocumentContent(uri: vscode.Uri): vscode.ProviderResult<string> {
        let syntax = activeSyntax()
        if (syntax == null) return
        if (uri.toString() == uris.syntaxTree.toString()) {
            return syntax.syntaxTree()
        }
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
    highlihgs: Array<[number, number, string]>
) {
    let byTag = {}
    for (let tag in decorations) {
        byTag[tag] = []
    }

    for (let [start, end, tag] of highlihgs) {
        if (!byTag[tag]) {
            console.log(`unknown tag ${tag}`)
            continue
        }
        let range = toVsRange(editor.document, [start, end])
        byTag[tag].push(range)
    }

    for (let tag in byTag) {
        let dec = decorations[tag]
        let ranges = byTag[tag]
        editor.setDecorations(dec, ranges)
    }
}

export function toVsRange(doc: vscode.TextDocument, range: [number, number]): vscode.Range {
    return new vscode.Range(
        doc.positionAt(range[0]),
        doc.positionAt(range[1]),
    )
}

function fromVsRange(doc: vscode.TextDocument, range: vscode.Range): [number, number] {
    return [doc.offsetAt(range.start), doc.offsetAt(range.end)]
}
