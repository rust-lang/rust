import * as vscode from 'vscode'
import { log } from 'util'

export function createPlugin(
    backend,
    fileExtension: string,
    disposables: vscode.Disposable[],
    doHighlighting: boolean = false,
    diganosticCollection: vscode.DiagnosticCollection | null = null
) {
    let uris = {
        syntaxTree: vscode.Uri.parse(`fall-${fileExtension}://syntaxtree`),
        metrics: vscode.Uri.parse(`fall-${fileExtension}://metrics`)
    }

    function updateActiveEditor() {
        let editor = vscode.window.activeTextEditor
        if (editor == null) return
        let file = currentFile()
        if (file == null) return
        if (doHighlighting) {
            setHighlights(editor, file.highlight())
        }
        if (diganosticCollection != null) {
            diganosticCollection.clear()
            diganosticCollection.set(
                editor.document.uri,
                file.diagnostics()
            )
        }
    }


    function currentFile(): EditorFile | null {
        let editor = vscode.window.activeTextEditor
        if (editor == null) return
        let doc = editor.document
        return getFile(doc)
    }

    vscode.window.onDidChangeActiveTextEditor(updateActiveEditor)
    let cmd = vscode.commands.registerCommand(`fall-${fileExtension}.applyContextAction`, (range, id) => {
        let file = currentFile()
        if (file == null) return
        return file.applyContextAction(range, id)
    })
    disposables.push(cmd)

    return {
        getFile: getFile,
        showSyntaxTree: () => {
            let file = currentFile()
            if (file == null) return
            return openDoc(uris.syntaxTree)
        },
        metrics: () => {
            let file = currentFile()
            if (file == null) return
            return openDoc(uris.metrics)
        },
        extendSelection: () => {
            let editor = vscode.window.activeTextEditor
            let file = currentFile()
            if (editor == null || file == null) return
            editor.selections = editor.selections.map((s) => {
                let range = file.extendSelection(s)
                return new vscode.Selection(range.start, range.end)
            })
        },
        documentSymbolsProvider: new DocumentSymbolProvider(getFile),
        documentFormattingEditProvider: new DocumentFormattingEditProvider(getFile),
        codeActionProvider: new CodeActionProvider(getFile, fileExtension)
    }
}


export interface FileStructureNode {
    name: string
    range: [number, number]
    children: [FileStructureNode]
}

export interface FallDiagnostic {
    range: [number, number]
    severity: string
    message: string
}

export class EditorFile {
    backend;
    imp;
    doc: vscode.TextDocument;

    constructor(backend, imp, doc: vscode.TextDocument) {
        this.backend = backend
        this.imp = imp
        this.doc = doc
    }

    metrics(): string { return this.call("metrics") }
    syntaxTree(): string { return this.call("syntaxTree") }
    extendSelection(range_: vscode.Range): vscode.Range | null {
        let range = fromVsRange(this.doc, range_)
        let exp = this.call("extendSelection", range)
        if (exp == null) return null
        return toVsRange(this.doc, exp)
    }

    structure(): Array<FileStructureNode> { return this.call("structure") }
    reformat(): Array<vscode.TextEdit> {
        let edits = this.call("reformat")
        return toVsEdits(this.doc, edits)
    }

    highlight(): Array<[[number, number], string]> { return this.call("highlight") }
    diagnostics(): Array<vscode.Diagnostic> {
        return this.call("diagnostics").map((d) => {
            let range = toVsRange(this.doc, d.range)
            let severity = d.severity == "Error"
                ? vscode.DiagnosticSeverity.Error
                : vscode.DiagnosticSeverity.Warning

            return new vscode.Diagnostic(range, d.message, severity)
        })
    }

    contextActions(range_: vscode.Range): Array<string> {
        let range = fromVsRange(this.doc, range_)
        let result = this.call("contextActions", range)
        return result
    }

    applyContextAction(range_: vscode.Range, id: string) {
        let range = fromVsRange(this.doc, range_)
        let edits = this.call("applyContextAction", range, id)
        let editor = vscode.window.activeTextEditor
        return editor.edit((builder) => {
            for (let op of edits) {
                builder.replace(toVsRange(this.doc, op.delete), op.insert)
            }
        })
    }

    call(method: string, ...args) {
        let result = this.backend[method](this.imp, ...args)
        return result
    }
}

function documentToFile(backend, fileExtension: string, disposables: vscode.Disposable[], onChange) {
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
        return doc.fileName.endsWith(`.${fileExtension}`)
    }

    vscode.workspace.onDidChangeTextDocument((event: vscode.TextDocumentChangeEvent) => {
        let doc = event.document
        if (!isKnownDoc(event.document)) return
        let tree = get(doc)
        if (event.contentChanges.length == 1 && tree) {
            let edits = event.contentChanges.map((change) => {
                let start = doc.offsetAt(change.range.start)
                return {
                    "delete": [start, start + change.rangeLength],
                    "insert": change.text
                }
            })
            update(doc, backend.edit(tree, edits))
            return
        }
        update(doc, null)
    }, null, disposables)

    vscode.workspace.onDidOpenTextDocument((doc: vscode.TextDocument) => {
        if (!isKnownDoc(doc)) return
        update(doc, backend.parse(doc.getText()))
    }, null, disposables)

    vscode.workspace.onDidCloseTextDocument((doc: vscode.TextDocument) => {
        update(doc, null)
    }, null, disposables)

    return (doc: vscode.TextDocument) => {
        if (!isKnownDoc(doc)) return null

        if (!get(doc)) {
            update(doc, backend.parse(doc.getText()))
        }
        let imp = get(doc)
        return new EditorFile(backend, imp, doc)
    }
}

export class DocumentSymbolProvider implements vscode.DocumentSymbolProvider {
    getFile: (doc: vscode.TextDocument) => EditorFile | null;
    constructor(getFile) {
        this.getFile = getFile
    }

    provideDocumentSymbols(document: vscode.TextDocument, token: vscode.CancellationToken) {
        let file = this.getFile(document)
        if (file == null) return null
        return file.structure().map((node) => {
            return new vscode.SymbolInformation(
                node.name,
                vscode.SymbolKind.Function,
                toVsRange(document, node.range),
                null,
                null
            )
        })
    }
}

export class DocumentFormattingEditProvider implements vscode.DocumentFormattingEditProvider {
    getFile: (doc: vscode.TextDocument) => EditorFile | null;
    constructor(getFile) { this.getFile = getFile }

    provideDocumentFormattingEdits(
        document: vscode.TextDocument,
        options: vscode.FormattingOptions,
        token: vscode.CancellationToken
    ): vscode.TextEdit[] {
        let file = this.getFile(document)
        if (file == null) return []
        return file.reformat()
    }
}

export class CodeActionProvider implements vscode.CodeActionProvider {
    fileExtension: string
    getFile: (doc: vscode.TextDocument) => EditorFile | null;
    constructor(getFile, fileExtension) {
        this.getFile = getFile
        this.fileExtension = fileExtension
    }

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): vscode.Command[] {
        let file = this.getFile(document)
        if (file == null) return
        let actions = file.contextActions(range)
        return actions.map((id) => {
            return {
                title: id,
                command: `fall-${this.fileExtension}.applyContextAction`,
                arguments: [range, id]
            }
        })
    }
}


export function toVsEdits(doc: vscode.TextDocument, edits): Array<vscode.TextEdit> {
    return edits.map((op) => vscode.TextEdit.replace(toVsRange(doc, op.delete), op.insert))
}

async function openDoc(uri: vscode.Uri) {
    let document = await vscode.workspace.openTextDocument(uri)
    vscode.window.showTextDocument(document, vscode.ViewColumn.Two, true)
}
