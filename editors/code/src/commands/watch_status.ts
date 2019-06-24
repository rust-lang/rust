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
            10
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

    private frame() {
        return spinnerFrames[(this.i = ++this.i % spinnerFrames.length)];
    }
}
