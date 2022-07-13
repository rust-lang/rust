import * as vscode from "vscode";
import * as lc from "vscode-languageclient/node";
import * as ra from "./lsp_ext";

import { Config } from "./config";
import { createClient } from "./client";
import { isRustEditor, RustEditor } from "./util";
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
    private constructor(
        readonly config: Config,
        private readonly extCtx: vscode.ExtensionContext,
        readonly client: lc.LanguageClient,
        readonly serverPath: string,
        readonly statusBar: vscode.StatusBarItem
    ) {}

    static async create(
        config: Config,
        extCtx: vscode.ExtensionContext,
        serverPath: string,
        workspace: Workspace
    ): Promise<Ctx> {
        const client = await createClient(serverPath, workspace, config.serverExtraEnv);

        const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
        extCtx.subscriptions.push(statusBar);
        statusBar.text = "rust-analyzer";
        statusBar.tooltip = "ready";
        statusBar.command = "rust-analyzer.analyzerStatus";
        statusBar.show();

        const res = new Ctx(config, extCtx, client, serverPath, statusBar);

        await client.start();
        client.onNotification(ra.serverStatus, (params) => res.setServerStatus(params));
        return res;
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
        this.pushCleanup(d);
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
                statusBar.tooltip += "\nClick to reload.";
                statusBar.command = "rust-analyzer.reloadWorkspace";
                statusBar.color = new vscode.ThemeColor("statusBarItem.errorForeground");
                statusBar.backgroundColor = new vscode.ThemeColor("statusBarItem.errorBackground");
                icon = "$(error) ";
                break;
        }
        if (!status.quiescent) icon = "$(sync~spin) ";
        statusBar.text = `${icon}rust-analyzer`;
    }

    pushCleanup(d: Disposable) {
        this.extCtx.subscriptions.push(d);
    }
}

export interface Disposable {
    dispose(): void;
}
export type Cmd = (...args: any[]) => unknown;
