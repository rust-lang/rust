import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import { Server } from './server';
import { Config } from './config';

export class Ctx {
    readonly config = new Config();
    private extCtx: vscode.ExtensionContext;

    constructor(extCtx: vscode.ExtensionContext) {
        this.extCtx = extCtx;
    }

    get client(): lc.LanguageClient {
        return Server.client;
    }

    get activeRustEditor(): vscode.TextEditor | undefined {
        const editor = vscode.window.activeTextEditor;
        return editor && editor.document.languageId === 'rust'
            ? editor
            : undefined;
    }

    registerCommand(name: string, factory: (ctx: Ctx) => Cmd) {
        const fullName = `rust-analyzer.${name}`;
        const cmd = factory(this);
        const d = vscode.commands.registerCommand(fullName, cmd);
        this.pushCleanup(d);
    }

    overrideCommand(name: string, factory: (ctx: Ctx) => Cmd) {
        const defaultCmd = `default:${name}`;
        const override = factory(this);
        const original = (...args: any[]) =>
            vscode.commands.executeCommand(defaultCmd, ...args);
        try {
            const d = vscode.commands.registerCommand(
                name,
                async (...args: any[]) => {
                    if (!(await override(...args))) {
                        return await original(...args);
                    }
                },
            );
            this.pushCleanup(d);
        } catch (_) {
            vscode.window.showWarningMessage(
                'Enhanced typing feature is disabled because of incompatibility with VIM extension, consider turning off rust-analyzer.enableEnhancedTyping: https://github.com/rust-analyzer/rust-analyzer/blob/master/docs/user/README.md#settings',
            );
        }
    }

    get subscriptions(): { dispose(): any }[] {
        return this.extCtx.subscriptions;
    }

    pushCleanup(d: { dispose(): any }) {
        this.extCtx.subscriptions.push(d);
    }

    async sendRequestWithRetry<R>(
        method: string,
        param: any,
        token?: vscode.CancellationToken,
    ): Promise<R> {
        await this.client.onReady();
        for (const delay of [2, 4, 6, 8, 10, null]) {
            try {
                return await (token ? this.client.sendRequest(method, param, token) : this.client.sendRequest(method, param));
            } catch (e) {
                if (
                    e.code === lc.ErrorCodes.ContentModified &&
                    delay !== null
                ) {
                    await sleep(10 * (1 << delay));
                    continue;
                }
                throw e;
            }
        }
        throw 'unreachable';
    }

    onNotification(method: string, handler: lc.GenericNotificationHandler) {
        this.client.onReady()
            .then(() => this.client.onNotification(method, handler))
    }
}

export type Cmd = (...args: any[]) => any;

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
