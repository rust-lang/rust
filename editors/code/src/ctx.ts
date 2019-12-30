import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';
import { Server } from './server';
import { Config } from './config';

export class Ctx {
    private extCtx: vscode.ExtensionContext;

    constructor(extCtx: vscode.ExtensionContext) {
        this.extCtx = extCtx;
    }

    get client(): lc.LanguageClient {
        return Server.client;
    }

    get config(): Config {
        return Server.config;
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
}

export type Cmd = (...args: any[]) => any;
