import * as vscode from "vscode";
import * as lc from "vscode-languageclient/node";
import * as ra from "./lsp_ext";

import { Config, substituteVariablesInEnv, substituteVSCodeVariables } from "./config";
import { createClient } from "./client";
import { isRustEditor, log, RustEditor } from "./util";
import { ServerStatusParams } from "./lsp_ext";

export type Workspace =
    | {
          kind: "Workspace Folder";
      }
    | {
          kind: "Detached Files";
          files: vscode.TextDocument[];
      };

export class Ctx {
    private client: lc.LanguageClient | undefined;
    readonly config: Config;
    serverPath: string;
    readonly statusBar: vscode.StatusBarItem;

    traceOutputChannel: vscode.OutputChannel | undefined;
    outputChannel: vscode.OutputChannel | undefined;

    serverOptions:
        | {
              run: lc.Executable;
              debug: lc.Executable;
          }
        | undefined;
    workspace: Workspace;

    constructor(
        readonly extCtx: vscode.ExtensionContext,
        config: Config,
        serverPath: string,
        workspace: Workspace
    ) {
        this.statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
        extCtx.subscriptions.push(this.statusBar);
        extCtx.subscriptions.push({
            dispose() {
                this.dispose();
            },
        });
        this.statusBar.text = "rust-analyzer";
        this.statusBar.tooltip = "ready";
        this.statusBar.command = "rust-analyzer.analyzerStatus";
        this.statusBar.show();
        this.serverPath = serverPath;
        this.config = config;
        this.workspace = workspace;
    }

    clientFetcher() {
        return {
            get client(): lc.LanguageClient | undefined {
                return this.client;
            },
        };
    }

    async getClient() {
        if (!this.traceOutputChannel) {
            this.traceOutputChannel = vscode.window.createOutputChannel(
                "Rust Analyzer Language Server Trace"
            );
        }
        if (!this.outputChannel) {
            this.outputChannel = vscode.window.createOutputChannel("Rust Analyzer Language Server");
        }
        if (!this.serverOptions) {
            log.info("Creating server options client");
            const newEnv = substituteVariablesInEnv(
                Object.assign({}, process.env, this.config.serverExtraEnv)
            );
            const run: lc.Executable = {
                command: this.serverPath,
                options: { env: newEnv },
            };
            this.serverOptions = {
                run,
                debug: run,
            };
        } else {
            this.serverOptions.run.command = this.serverPath;
            this.serverOptions.debug.command = this.serverPath;
        }
        if (!this.client) {
            log.info("Creating language client");
            let rawInitializationOptions = vscode.workspace.getConfiguration("rust-analyzer");

            if (this.workspace.kind === "Detached Files") {
                rawInitializationOptions = {
                    detachedFiles: this.workspace.files.map((file) => file.uri.fsPath),
                    ...rawInitializationOptions,
                };
            }

            const initializationOptions = substituteVSCodeVariables(rawInitializationOptions);

            this.client = await createClient(
                this.traceOutputChannel,
                this.outputChannel,
                initializationOptions,
                this.serverOptions
            );
            this.client.onNotification(ra.serverStatus, (params) => this.setServerStatus(params));
        }
        return this.client;
    }

    async activate() {
        log.info("Activating language client");
        const client = await this.getClient();
        await client.start();
        return client;
    }

    async deactivate() {
        log.info("Deactivating language client");
        await this.client?.stop();
    }

    async disposeClient() {
        log.info("Deactivating language client");
        await this.client?.dispose();
        this.client = undefined;
    }

    get activeRustEditor(): RustEditor | undefined {
        const editor = vscode.window.activeTextEditor;
        return editor && isRustEditor(editor) ? editor : undefined;
    }

    get visibleRustEditors(): RustEditor[] {
        return vscode.window.visibleTextEditors.filter(isRustEditor);
    }

    registerCommand(name: string, factory: (ctx: Ctx) => Cmd) {
        const fullName = `rust-analyzer.${name}`;
        const cmd = factory(this);
        const d = vscode.commands.registerCommand(fullName, cmd);
        this.pushExtCleanup(d);
    }

    get extensionPath(): string {
        return this.extCtx.extensionPath;
    }

    get globalState(): vscode.Memento {
        return this.extCtx.globalState;
    }

    get subscriptions(): Disposable[] {
        return this.extCtx.subscriptions;
    }

    setServerStatus(status: ServerStatusParams) {
        let icon = "";
        const statusBar = this.statusBar;
        switch (status.health) {
            case "ok":
                statusBar.tooltip = status.message ?? "Ready";
                statusBar.command = undefined;
                statusBar.color = undefined;
                statusBar.backgroundColor = undefined;
                break;
            case "warning":
                statusBar.tooltip =
                    (status.message ? status.message + "\n" : "") + "Click to reload.";

                statusBar.command = "rust-analyzer.reloadWorkspace";
                statusBar.color = new vscode.ThemeColor("statusBarItem.warningForeground");
                statusBar.backgroundColor = new vscode.ThemeColor(
                    "statusBarItem.warningBackground"
                );
                icon = "$(warning) ";
                break;
            case "error":
                statusBar.tooltip =
                    (status.message ? status.message + "\n" : "") + "Click to reload.";

                statusBar.command = "rust-analyzer.reloadWorkspace";
                statusBar.color = new vscode.ThemeColor("statusBarItem.errorForeground");
                statusBar.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
                icon = "$(error) ";
                break;
        }
        if (!status.quiescent) icon = "$(sync~spin) ";
        statusBar.text = `${icon}rust-analyzer`;
    }

    pushExtCleanup(d: Disposable) {
        this.extCtx.subscriptions.push(d);
    }
}

export interface Disposable {
    dispose(): void;
}
export type Cmd = (...args: any[]) => unknown;
