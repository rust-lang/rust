import * as child_process from 'child_process';
import * as path from 'path';
import * as vscode from 'vscode';
import { terminate } from '../utils/processes';
import { StatusDisplay } from './watch_status';


export class CargoWatchProvider {
    private diagnosticCollection?: vscode.DiagnosticCollection;
    private cargoProcess?: child_process.ChildProcess;
    private outBuffer: string = '';
    private statusDisplay?: StatusDisplay;

    public activate(subscriptions: vscode.Disposable[]) {
        subscriptions.push(this);
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection(
            'rustc'
        );

        this.statusDisplay = new StatusDisplay(subscriptions);

        // Start the cargo watch with json message
        this.cargoProcess = child_process.spawn(
            'cargo',
            ['watch', '-x', '\"check --message-format json\"'],
            {
                stdio: ['ignore', 'pipe', 'pipe'],
                cwd: vscode.workspace.rootPath,
                windowsVerbatimArguments: true,
            }
        );

        this.cargoProcess.stdout.on('data', (s: string) => {
            this.processOutput(s);
            console.log(s);
        });

        this.cargoProcess.stderr.on('data', (s: string) => {
            console.error('Error on cargo watch : ' + s);
        });

        this.cargoProcess.on('error', (err: Error) => {
            console.error('Error on spawn cargo process : ' + err);
        });
    }

    public dispose(): void {
        if (this.diagnosticCollection) {
            this.diagnosticCollection.clear();
            this.diagnosticCollection.dispose();
        }

        if (this.cargoProcess) {
            this.cargoProcess.kill();
            terminate(this.cargoProcess);
        }
    }

    private parseLine(line: string) {
        if (line.startsWith('[Running')) {
            this.diagnosticCollection!.clear();
            this.statusDisplay!.show();
        }

        if (line.startsWith('[Finished running')) {
            this.statusDisplay!.hide();
        }

        function getLevel(s: string): vscode.DiagnosticSeverity {
            if (s === 'error') {
                return vscode.DiagnosticSeverity.Error;
            }

            if (s.startsWith('warn')) {
                return vscode.DiagnosticSeverity.Warning;
            }

            return vscode.DiagnosticSeverity.Information;
        }

        // cargo-watch itself output non json format
        // Ignore these lines
        let data = null;
        try {
            data = JSON.parse(line.trim());
        } catch (error) {
            return;
        }

        // Only handle compiler-message now
        if (data.reason !== 'compiler-message') {
            return;
        }

        let spans: any[] = data.message.spans;
        spans = spans.filter(o => o.is_primary);

        // We only handle primary span right now.
        if (spans.length > 0) {
            const o = spans[0];

            const rendered = data.message.rendered;
            const level = getLevel(data.message.level);
            const range = new vscode.Range(
                new vscode.Position(o.line_start - 1, o.column_start - 1),
                new vscode.Position(o.line_end - 1, o.column_end - 1)
            );

            const fileName = path.join(vscode.workspace.rootPath!, o.file_name);
            const diagnostic = new vscode.Diagnostic(range, rendered, level);

            diagnostic.source = 'rustc';
            diagnostic.code = data.message.code.code;
            diagnostic.relatedInformation = [];

            const fileUrl = vscode.Uri.file(fileName!);

            const diagnostics: vscode.Diagnostic[] = [
                ...(this.diagnosticCollection!.get(fileUrl) || [])
            ];
            diagnostics.push(diagnostic);

            this.diagnosticCollection!.set(fileUrl, diagnostics);
        }
    }

    private processOutput(chunk: string) {
        // The stdout is not line based, convert it to line based for proceess.
        this.outBuffer += chunk;
        let eolIndex = this.outBuffer.indexOf('\n');
        while (eolIndex >= 0) {
            // line includes the EOL
            const line = this.outBuffer.slice(0, eolIndex + 1);
            this.parseLine(line);
            this.outBuffer = this.outBuffer.slice(eolIndex + 1);

            eolIndex = this.outBuffer.indexOf('\n');
        }
    }
}
