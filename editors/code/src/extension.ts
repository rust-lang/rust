import { exec, spawn } from 'child_process';
import * as util from 'util';
import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import * as commands from './commands';
import { autoCargoWatchTask, createTask } from './commands/runnables';
import { SyntaxTreeContentProvider } from './commands/syntaxTree';
import * as events from './events';
import * as notifications from './notifications';
import { Server } from './server';
import { TextDecoder } from 'util';

export function activate(context: vscode.ExtensionContext) {
    function disposeOnDeactivation(disposable: vscode.Disposable) {
        context.subscriptions.push(disposable);
    }

    function registerCommand(name: string, f: any) {
        disposeOnDeactivation(vscode.commands.registerCommand(name, f));
    }
    function overrideCommand(
        name: string,
        f: (...args: any[]) => Promise<boolean>
    ) {
        const defaultCmd = `default:${name}`;
        const original = (...args: any[]) =>
            vscode.commands.executeCommand(defaultCmd, ...args);

        try {
            registerCommand(name, async (...args: any[]) => {
                const editor = vscode.window.activeTextEditor;
                if (
                    !editor ||
                    !editor.document ||
                    editor.document.languageId !== 'rust'
                ) {
                    return await original(...args);
                }
                if (!(await f(...args))) {
                    return await original(...args);
                }
            });
        } catch (_) {
            vscode.window.showWarningMessage(
                'Enhanced typing feature is disabled because of incompatibility with VIM extension'
            );
        }
    }

    // Commands are requests from vscode to the language server
    registerCommand(
        'rust-analyzer.analyzerStatus',
        commands.analyzerStatus.makeCommand(context)
    );
    registerCommand('rust-analyzer.collectGarbage', () =>
        Server.client.sendRequest<null>('rust-analyzer/collectGarbage', null)
    );
    registerCommand(
        'rust-analyzer.extendSelection',
        commands.extendSelection.handle
    );
    registerCommand(
        'rust-analyzer.matchingBrace',
        commands.matchingBrace.handle
    );
    registerCommand('rust-analyzer.joinLines', commands.joinLines.handle);
    registerCommand('rust-analyzer.parentModule', commands.parentModule.handle);
    registerCommand('rust-analyzer.run', commands.runnables.handle);
    // Unlike the above this does not send requests to the language server
    registerCommand('rust-analyzer.runSingle', commands.runnables.handleSingle);
    registerCommand(
        'rust-analyzer.applySourceChange',
        commands.applySourceChange.handle
    );
    registerCommand(
        'rust-analyzer.showReferences',
        (uri: string, position: lc.Position, locations: lc.Location[]) => {
            vscode.commands.executeCommand(
                'editor.action.showReferences',
                vscode.Uri.parse(uri),
                Server.client.protocol2CodeConverter.asPosition(position),
                locations.map(Server.client.protocol2CodeConverter.asLocation)
            );
        }
    );

    if (Server.config.enableEnhancedTyping) {
        overrideCommand('type', commands.onEnter.handle);
    }

    // Notifications are events triggered by the language server
    const allNotifications: Iterable<
        [string, lc.GenericNotificationHandler]
    > = [
            [
                'rust-analyzer/publishDecorations',
                notifications.publishDecorations.handle
            ]
        ];
    const syntaxTreeContentProvider = new SyntaxTreeContentProvider();

    // The events below are plain old javascript events, triggered and handled by vscode
    vscode.window.onDidChangeActiveTextEditor(
        events.changeActiveTextEditor.makeHandler(syntaxTreeContentProvider)
    );

    disposeOnDeactivation(
        vscode.workspace.registerTextDocumentContentProvider(
            'rust-analyzer',
            syntaxTreeContentProvider
        )
    );

    registerCommand(
        'rust-analyzer.syntaxTree',
        commands.syntaxTree.createHandle(syntaxTreeContentProvider)
    );

    vscode.workspace.onDidChangeTextDocument(
        events.changeTextDocument.createHandler(syntaxTreeContentProvider),
        null,
        context.subscriptions
    );

    // Attempts to run `cargo watch`, which provides inline diagnostics on save
    askToCargoWatch();

    // Start the language server, finally!
    Server.start(allNotifications);
}

export function deactivate(): Thenable<void> {
    if (!Server.client) {
        return Promise.resolve();
    }
    return Server.client.stop();
}

async function askToCargoWatch() {
    const watch = await vscode.window.showInformationMessage(
        'Start watching changes with cargo? (Executes `cargo watch`, provides inline diagnostics)',
        'yes',
        'no'
    );
    if (watch === 'no') {
        return;
    }

    const { stderr } = await util.promisify(exec)('cargo watch --version').catch(e => e);
    if (stderr.includes('no such subcommand: `watch`')) {
        const msg = 'The `cargo-watch` subcommand is not installed. Install? (takes ~1-2 minutes)';
        const install = await vscode.window.showInformationMessage(msg, 'yes', 'no');
        if (install === 'no') {
            return;
        }

        try {
            // await vscode.tasks.executeTask(createTask({label: '', bin: 'cargo', args: ['install', 'cargo-watch'], env: {}}));

            const channel = vscode.window.createOutputChannel('cargo-watch');
            channel.show(false);
            const sup = spawn('cargo', ['install', 'cargo-watch']);
            sup.stderr.on('data', chunk => {
                const output = new TextDecoder().decode(chunk);
                channel.append(output);
            });
            await new Promise((resolve, reject) => {
                sup.on('close', (code, signal) => {
                    if (code === 0) {
                        resolve(code);
                    } else {
                        reject(code);
                    }
                });
            });
            channel.dispose();
        } catch (err) {
            vscode.window.showErrorMessage(
                `Couldn't install \`cargo-watch\`: ${err.message}`
            );
            return;
        }
    }

    vscode.tasks.executeTask(autoCargoWatchTask);
}
