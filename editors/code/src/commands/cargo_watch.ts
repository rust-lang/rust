import * as child_process from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { Server } from '../server';
import { terminate } from '../utils/processes';
import { LineBuffer } from './line_buffer';
import { StatusDisplay } from './watch_status';

export function registerCargoWatchProvider(
    subscriptions: vscode.Disposable[]
): CargoWatchProvider | undefined {
    let cargoExists = false;
    const cargoTomlFile = path.join(vscode.workspace.rootPath!, 'Cargo.toml');
    // Check if the working directory is valid cargo root path
    try {
        if (fs.existsSync(cargoTomlFile)) {
            cargoExists = true;
        }
    } catch (err) {
        cargoExists = false;
    }

    if (!cargoExists) {
        vscode.window.showErrorMessage(
            `Couldn\'t find \'Cargo.toml\' in ${cargoTomlFile}`
        );
        return;
    }

    const provider = new CargoWatchProvider();
    subscriptions.push(provider);
    return provider;
}

export class CargoWatchProvider implements vscode.Disposable {
    private readonly diagnosticCollection: vscode.DiagnosticCollection;
    private readonly statusDisplay: StatusDisplay;
    private readonly outputChannel: vscode.OutputChannel;
    private cargoProcess?: child_process.ChildProcess;

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection(
            'rustc'
        );
        this.statusDisplay = new StatusDisplay();
        this.outputChannel = vscode.window.createOutputChannel(
            'Cargo Watch Trace'
        );
    }

    public start() {
        if (this.cargoProcess) {
            vscode.window.showInformationMessage(
                'Cargo Watch is already running'
            );
            return;
        }

        let args = 'check --all-targets --message-format json';
        if (Server.config.cargoWatchOptions.checkArguments.length > 0) {
            // Excape the double quote string:
            args += ' ' + Server.config.cargoWatchOptions.checkArguments;
        }
        // Windows handles arguments differently than the unix-likes, so we need to wrap the args in double quotes
        if (process.platform === 'win32') {
            args = '"' + args + '"';
        }

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
                try {
                    this.parseLine(line);
                } catch (err) {
                    this.logError(`Failed to parse: ${err}, content : ${line}`);
                }
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

    public stop() {
        if (this.cargoProcess) {
            this.cargoProcess.kill();
            terminate(this.cargoProcess);
            this.cargoProcess = undefined;
        } else {
            vscode.window.showInformationMessage('Cargo Watch is not running');
        }
    }

    public dispose(): void {
        this.stop();

        this.diagnosticCollection.clear();
        this.diagnosticCollection.dispose();
        this.outputChannel.dispose();
        this.statusDisplay.dispose();
    }

    private logInfo(line: string) {
        if (Server.config.cargoWatchOptions.trace === 'verbose') {
            this.outputChannel.append(line);
        }
    }

    private logError(line: string) {
        if (
            Server.config.cargoWatchOptions.trace === 'error' ||
            Server.config.cargoWatchOptions.trace === 'verbose'
        ) {
            this.outputChannel.append(line);
        }
    }

    private parseLine(line: string) {
        if (line.startsWith('[Running')) {
            this.diagnosticCollection.clear();
            this.statusDisplay.show();
        }

        if (line.startsWith('[Finished running')) {
            this.statusDisplay.hide();
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

        // Reference:
        // https://github.com/rust-lang/rust/blob/master/src/libsyntax/json.rs
        interface RustDiagnosticSpan {
            line_start: number;
            line_end: number;
            column_start: number;
            column_end: number;
            is_primary: boolean;
            file_name: string;
        }

        interface RustDiagnostic {
            spans: RustDiagnosticSpan[];
            rendered: string;
            level: string;
            code?: {
                code: string;
            };
        }

        interface CargoArtifact {
            reason: string;
            package_id: string;
        }

        // https://github.com/rust-lang/cargo/blob/master/src/cargo/util/machine_message.rs
        interface CargoMessage {
            reason: string;
            package_id: string;
            message: RustDiagnostic;
        }

        // cargo-watch itself output non json format
        // Ignore these lines
        let data: CargoMessage;
        try {
            data = JSON.parse(line.trim());
        } catch (error) {
            this.logError(`Fail to parse to json : { ${error} }`);
            return;
        }

        if (data.reason === 'compiler-artifact') {
            const msg = data as CargoArtifact;

            // The format of the package_id is "{name} {version} ({source_id})",
            // https://github.com/rust-lang/cargo/blob/37ad03f86e895bb80b474c1c088322634f4725f5/src/cargo/core/package_id.rs#L53
            this.statusDisplay.packageName = msg.package_id.split(' ')[0];
        } else if (data.reason === 'compiler-message') {
            const msg = data.message as RustDiagnostic;

            const spans = msg.spans.filter(o => o.is_primary);

            // We only handle primary span right now.
            if (spans.length > 0) {
                const o = spans[0];

                const rendered = msg.rendered;
                const level = getLevel(msg.level);
                const range = new vscode.Range(
                    new vscode.Position(o.line_start - 1, o.column_start - 1),
                    new vscode.Position(o.line_end - 1, o.column_end - 1)
                );

                const fileName = path.join(
                    vscode.workspace.rootPath!,
                    o.file_name
                );
                const diagnostic = new vscode.Diagnostic(
                    range,
                    rendered,
                    level
                );

                diagnostic.source = 'rustc';
                diagnostic.code = msg.code ? msg.code.code : undefined;
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
}
