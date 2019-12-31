import * as vscode from 'vscode';

import { Ctx } from './ctx';

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

export function activateStatusDisplay(ctx: Ctx) {
    const statusDisplay = new StatusDisplay(ctx.config.cargoWatchOptions.command);
    ctx.pushCleanup(statusDisplay);
    ctx.onDidRestart(client => {
        client.onNotification('$/progress', params => statusDisplay.handleProgressNotification(params));
    });
}

class StatusDisplay implements vscode.Disposable {
    packageName?: string;

    private i = 0;
    private statusBarItem: vscode.StatusBarItem;
    private command: string;
    private timer?: NodeJS.Timeout;

    constructor(command: string) {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            10,
        );
        this.command = command;
        this.statusBarItem.hide();
    }

    show() {
        this.packageName = undefined;

        this.timer =
            this.timer ||
            setInterval(() => {
                if (this.packageName) {
                    this.statusBarItem!.text = `${this.frame()} cargo ${this.command} [${this.packageName}]`;
                } else {
                    this.statusBarItem!.text = `${this.frame()} cargo ${this.command}`;
                }
            }, 300);

        this.statusBarItem.show();
    }

    hide() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = undefined;
        }

        this.statusBarItem.hide();
    }

    dispose() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = undefined;
        }

        this.statusBarItem.dispose();
    }

    handleProgressNotification(params: ProgressParams) {
        const { token, value } = params;
        if (token !== 'rustAnalyzer/cargoWatcher') {
            return;
        }

        switch (value.kind) {
            case 'begin':
                this.show();
                break;

            case 'report':
                if (value.message) {
                    this.packageName = value.message;
                }
                break;

            case 'end':
                this.hide();
                break;
        }
    }

    private frame() {
        return spinnerFrames[(this.i = ++this.i % spinnerFrames.length)];
    }
}

// FIXME: Replace this once vscode-languageclient is updated to LSP 3.15
interface ProgressParams {
    token: string;
    value: WorkDoneProgress;
}

enum WorkDoneProgressKind {
    Begin = 'begin',
    Report = 'report',
    End = 'end',
}

interface WorkDoneProgress {
    kind: WorkDoneProgressKind;
    message?: string;
    cancelable?: boolean;
    percentage?: string;
}
