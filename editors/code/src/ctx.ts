import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Config } from './config';
import { createClient } from './client';
import { isRustEditor, RustEditor } from './util';

export class Ctx {
    private constructor(
        readonly config: Config,
        private readonly extCtx: vscode.ExtensionContext,
        readonly client: lc.LanguageClient
    ) {

    }

    static async create(config: Config, extCtx: vscode.ExtensionContext, serverPath: string): Promise<Ctx> {
        const client = await createClient(config, serverPath);
        const res = new Ctx(config, extCtx, client);
        res.pushCleanup(client.start());
        await client.onReady();
        return res;
    }

    get activeRustEditor(): RustEditor | undefined {
        const editor = vscode.window.activeTextEditor;
        return editor && isRustEditor(editor)
            ? editor
            : undefined;
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

    get globalState(): vscode.Memento {
        return this.extCtx.globalState;
    }

    get subscriptions(): Disposable[] {
        return this.extCtx.subscriptions;
    }

    pushCleanup(d: Disposable) {
        this.extCtx.subscriptions.push(d);
    }
}

export interface Disposable {
    dispose(): void;
}
export type Cmd = (...args: any[]) => unknown;
