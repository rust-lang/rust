import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import { strict as assert } from "assert";

import { Config } from './config';
import { createClient } from './client';

export class Ctx {
    readonly config: Config;
    // Because we have "reload server" action, various listeners **will** face a
    // situation where the client is not ready yet, and should be prepared to
    // deal with it.
    //
    // Ideally, this should be replaced with async getter though.
    // FIXME: this actually needs syncronization of some kind (check how
    // vscode deals with `deactivate()` call when extension has some work scheduled
    // on the event loop to get a better picture of what we can do here)
    client: lc.LanguageClient | null = null;
    private extCtx: vscode.ExtensionContext;
    private onStartHooks: Array<(client: lc.LanguageClient) => void> = [];

    constructor(extCtx: vscode.ExtensionContext) {
        this.config = new Config(extCtx);
        this.extCtx = extCtx;
    }

    async startServer() {
        assert(this.client == null);

        const client = await createClient(this.config);
        if (!client) {
            throw new Error(
                "Rust Analyzer Language Server is not available. " +
                "Please, ensure its [proper installation](https://github.com/rust-analyzer/rust-analyzer/tree/master/docs/user#vs-code)."
            );
        }

        this.pushCleanup(client.start());
        await client.onReady();

        this.client = client;
        for (const hook of this.onStartHooks) {
            hook(client);
        }
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

    get globalState(): vscode.Memento {
        return this.extCtx.globalState;
    }

    get subscriptions(): Disposable[] {
        return this.extCtx.subscriptions;
    }

    pushCleanup(d: Disposable) {
        this.extCtx.subscriptions.push(d);
    }

    onStart(hook: (client: lc.LanguageClient) => void) {
        const client = this.client;
        if (client == null) {
            this.onStartHooks.push(hook);
        } else {
            hook(client)
        }
    }
}

export interface Disposable {
    dispose(): void;
}
export type Cmd = (...args: any[]) => unknown;

export async function sendRequestWithRetry<R>(
    client: lc.LanguageClient,
    method: string,
    param: unknown,
    token?: vscode.CancellationToken,
): Promise<R> {
    for (const delay of [2, 4, 6, 8, 10, null]) {
        try {
            return await (token ? client.sendRequest(method, param, token) : client.sendRequest(method, param));
        } catch (err) {
            if (delay === null || err.code !== lc.ErrorCodes.ContentModified) {
                throw err;
            }
            await sleep(10 * (1 << delay));
        }
    }
    throw 'unreachable';
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
