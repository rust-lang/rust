import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import { Server } from './server';

export class Ctx {
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

    pushCleanup(d: { dispose(): any }) {
        this.extCtx.subscriptions.push(d);
    }
}

export type Cmd = (...args: any[]) => any;
