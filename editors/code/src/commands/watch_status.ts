import * as vscode from 'vscode';

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

export class StatusDisplay implements vscode.Disposable {
    public packageName?: string;

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

    public show() {
        this.packageName = undefined;

        this.timer =
            this.timer ||
            setInterval(() => {
                if (this.packageName) {
                    this.statusBarItem!.text = `cargo ${this.command} [${
                        this.packageName
                    }] ${this.frame()}`;
                } else {
                    this.statusBarItem!.text = `cargo ${
                        this.command
                    } ${this.frame()}`;
                }
            }, 300);

        this.statusBarItem.show();
    }

    public hide() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = undefined;
        }

        this.statusBarItem.hide();
    }

    public dispose() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = undefined;
        }

        this.statusBarItem.dispose();
    }

    public handleProgressNotification(params: ProgressParams) {
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
