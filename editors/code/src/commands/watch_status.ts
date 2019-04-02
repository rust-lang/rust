import * as vscode from 'vscode';

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

export class StatusDisplay {
    public packageName?: string;

    private i = 0;
    private statusBarItem: vscode.StatusBarItem;
    private timer?: NodeJS.Timeout;

    constructor(subscriptions: vscode.Disposable[]) {
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            10
        );
        subscriptions.push(this.statusBarItem);
        this.statusBarItem.hide();
    }

    public show() {
        this.packageName = undefined;

        this.timer =
            this.timer ||
            setInterval(() => {
                if (this.packageName) {
                    this.statusBarItem!.text = `cargo check [${
                        this.packageName
                    }] ${this.frame()}`;
                } else {
                    this.statusBarItem!.text = `cargo check ${this.frame()}`;
                }
            }, 300);

        this.statusBarItem!.show();
    }

    public hide() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = undefined;
        }

        this.statusBarItem!.hide();
    }

    private frame() {
        return spinnerFrames[(this.i = ++this.i % spinnerFrames.length)];
    }
}
