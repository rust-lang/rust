import * as child_process from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

import { Server } from '../server';
import { terminate } from '../utils/processes';
import {
    mapRustDiagnosticToVsCode,
    RustDiagnostic
} from '../utils/rust_diagnostics';
import {
    areCodeActionsEqual,
    areDiagnosticsEqual
} from '../utils/vscode_diagnostics';
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

export class CargoWatchProvider
    implements vscode.Disposable, vscode.CodeActionProvider {
    private readonly diagnosticCollection: vscode.DiagnosticCollection;
    private readonly statusDisplay: StatusDisplay;
    private readonly outputChannel: vscode.OutputChannel;

    private codeActions: {
        [fileUri: string]: vscode.CodeAction[];
    };
    private readonly codeActionDispose: vscode.Disposable;

    private cargoProcess?: child_process.ChildProcess;

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection(
            'rustc'
        );
        this.statusDisplay = new StatusDisplay(
            Server.config.cargoWatchOptions.command
        );
        this.outputChannel = vscode.window.createOutputChannel(
            'Cargo Watch Trace'
        );

        // Register code actions for rustc's suggested fixes
        this.codeActions = {};
        this.codeActionDispose = vscode.languages.registerCodeActionsProvider(
            [{ scheme: 'file', language: 'rust' }],
            this,
            {
                providedCodeActionKinds: [vscode.CodeActionKind.QuickFix]
            }
        );
    }

    public start() {
        if (this.cargoProcess) {
            vscode.window.showInformationMessage(
                'Cargo Watch is already running'
            );
            return;
        }

        let args =
            Server.config.cargoWatchOptions.command +
            ' --all-targets --message-format json';
        if (Server.config.cargoWatchOptions.command.length > 0) {
            // Excape the double quote string:
            args += ' ' + Server.config.cargoWatchOptions.arguments;
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
        this.codeActionDispose.dispose();
    }

    public provideCodeActions(
        document: vscode.TextDocument
    ): vscode.ProviderResult<Array<vscode.Command | vscode.CodeAction>> {
        const documentActions = this.codeActions[document.uri.toString()];
        return documentActions || [];
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
            this.codeActions = {};
            this.statusDisplay.show();
        }

        if (line.startsWith('[Finished running')) {
            this.statusDisplay.hide();
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

            const mapResult = mapRustDiagnosticToVsCode(msg);
            if (!mapResult) {
                return;
            }

            const { location, diagnostic, codeActions } = mapResult;
            const fileUri = location.uri;

            const diagnostics: vscode.Diagnostic[] = [
                ...(this.diagnosticCollection!.get(fileUri) || [])
            ];

            // If we're building multiple targets it's possible we've already seen this diagnostic
            const isDuplicate = diagnostics.some(d =>
                areDiagnosticsEqual(d, diagnostic)
            );

            if (isDuplicate) {
                return;
            }

            diagnostics.push(diagnostic);
            this.diagnosticCollection!.set(fileUri, diagnostics);

            if (codeActions.length) {
                const fileUriString = fileUri.toString();
                const existingActions = this.codeActions[fileUriString] || [];

                for (const newAction of codeActions) {
                    const existingAction = existingActions.find(existing =>
                        areCodeActionsEqual(existing, newAction)
                    );

                    if (existingAction) {
                        if (!existingAction.diagnostics) {
                            existingAction.diagnostics = [];
                        }
                        // This action also applies to this diagnostic
                        existingAction.diagnostics.push(diagnostic);
                    } else {
                        newAction.diagnostics = [diagnostic];
                        existingActions.push(newAction);
                    }
                }

                // Have VsCode query us for the code actions
                this.codeActions[fileUriString] = existingActions;
                vscode.commands.executeCommand(
                    'vscode.executeCodeActionProvider',
                    fileUri,
                    diagnostic.range
                );
            }
        }
    }
}
