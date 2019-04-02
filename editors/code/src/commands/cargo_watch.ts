import * as child_process from 'child_process';
import * as path from 'path';
import * as vscode from 'vscode';
import { Server } from '../server';
import { terminate } from '../utils/processes';
import { LineBuffer } from './line_buffer';
import { StatusDisplay } from './watch_status';

export class CargoWatchProvider {
    private diagnosticCollection?: vscode.DiagnosticCollection;
    private cargoProcess?: child_process.ChildProcess;
    private outBuffer: string = '';
    private statusDisplay?: StatusDisplay;
    private outputChannel?: vscode.OutputChannel;

    public activate(subscriptions: vscode.Disposable[]) {
        subscriptions.push(this);
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection(
            'rustc'
        );

        this.statusDisplay = new StatusDisplay(subscriptions);
        this.outputChannel = vscode.window.createOutputChannel(
            'Cargo Watch Trace'
        );

        let args = '"check --message-format json';
        if (Server.config.cargoWatchOptions.checkArguments.length > 0) {
            // Excape the double quote string:
            args += ' ' + Server.config.cargoWatchOptions.checkArguments;
        }
        args += '"';

        // Start the cargo watch with json message
        this.cargoProcess = child_process.spawn(
            'cargo',
            ['watch', '-x', args],
            {
                stdio: ['ignore', 'pipe', 'pipe'],
                cwd: vscode.workspace.rootPath,
                windowsVerbatimArguments: true
            }
        );

        const stdoutData = new LineBuffer();
        this.cargoProcess.stdout.on('data', (s: string) => {
            stdoutData.processOutput(s, line => {
                this.logInfo(line);
                this.parseLine(line);
            });
        });

        const stderrData = new LineBuffer();
        this.cargoProcess.stderr.on('data', (s: string) => {
            stderrData.processOutput(s, line => {
                this.logError('Error on cargo-watch : {\n' + line + '}\n');
            });
        });

        this.cargoProcess.on('error', (err: Error) => {
            this.logError(
                'Error on cargo-watch process : {\n' + err.message + '}\n'
            );
        });

        this.logInfo('cargo-watch started.');
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

        if (this.outputChannel) {
            this.outputChannel.dispose();
        }
    }

    private logInfo(line: string) {
        if (Server.config.cargoWatchOptions.trace === 'verbose') {
            this.outputChannel!.append(line);
        }
    }

    private logError(line: string) {
        if (
            Server.config.cargoWatchOptions.trace === 'error' ||
            Server.config.cargoWatchOptions.trace === 'verbose'
        ) {
            this.outputChannel!.append(line);
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

        interface ErrorSpan {
            line_start: number;
            line_end: number;
            column_start: number;
            column_end: number;
        }

        interface ErrorMessage {
            reason: string;
            message: {
                spans: ErrorSpan[];
                rendered: string;
                level: string;
                code?: {
                    code: string;
                };
            };
        }

        // cargo-watch itself output non json format
        // Ignore these lines
        let data: ErrorMessage;
        try {
            data = JSON.parse(line.trim());
        } catch (error) {
            this.logError(`Fail to pass to json : { ${error} }`);
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
            diagnostic.code = data.message.code
                ? data.message.code.code
                : undefined;
            diagnostic.relatedInformation = [];

            const fileUrl = vscode.Uri.file(fileName!);

            const diagnostics: vscode.Diagnostic[] = [
                ...(this.diagnosticCollection!.get(fileUrl) || [])
            ];
            diagnostics.push(diagnostic);

            this.diagnosticCollection!.set(fileUrl, diagnostics);
        }
    }
}
