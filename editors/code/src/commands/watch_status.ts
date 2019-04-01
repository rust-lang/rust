import * as vscode from 'vscode';

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

export class StatusDisplay {
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
        this.timer =
            this.timer ||
            setInterval(() => {
                this.statusBarItem!.text = 'cargo check ' + this.frame();
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
