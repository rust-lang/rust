import * as vscode from 'vscode';

import { WorkDoneProgress, WorkDoneProgressBegin, WorkDoneProgressReport, WorkDoneProgressEnd, Disposable } from 'vscode-languageclient';

import { Ctx } from './ctx';

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

export function activateStatusDisplay(ctx: Ctx) {
    const statusDisplay = new StatusDisplay(ctx.config.cargoWatchOptions.command);
    ctx.pushCleanup(statusDisplay);
    const client = ctx.client;
    if (client != null) {
        ctx.pushCleanup(client.onProgress(
            WorkDoneProgress.type,
            'rustAnalyzer/cargoWatcher',
            params => statusDisplay.handleProgressNotification(params)
        ));
    }
}

class StatusDisplay implements Disposable {
    packageName?: string;

    private i: number = 0;
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
                this.tick();
                this.refreshLabel();
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

    refreshLabel() {
        if (this.packageName) {
            this.statusBarItem.text = `${spinnerFrames[this.i]} cargo ${this.command} [${this.packageName}]`;
        } else {
            this.statusBarItem.text = `${spinnerFrames[this.i]} cargo ${this.command}`;
        }
    }

    handleProgressNotification(params: WorkDoneProgressBegin | WorkDoneProgressReport | WorkDoneProgressEnd) {
        switch (params.kind) {
            case 'begin':
                this.show();
                break;

            case 'report':
                if (params.message) {
                    this.packageName = params.message;
                    this.refreshLabel();
                }
                break;

            case 'end':
                this.hide();
                break;
        }
    }

    private tick() {
        this.i = (this.i + 1) % spinnerFrames.length;
    }
}
