import { Ctx, Cmd } from '../ctx';
import { applySourceChange, SourceChange } from '../source_change';
import * as vscode from 'vscode';

export function ssr(ctx: Ctx): Cmd {
    return async () => {
        const client = ctx.client;
        if (!client) return;

        const options: vscode.InputBoxOptions = {
            placeHolder: "foo($a:expr, $b:expr) ==>> bar($a, foo($b))",
            prompt: "Enter request",
            validateInput: (x: string) => {
                if (x.includes('==>>')) {
                    return null;
                }
                return "Enter request: pattern ==>> template";
            }
        };
        const request = await vscode.window.showInputBox(options);

        if (!request) return;

        const ssrRequest: SsrRequest = { arg: request };
        const change = await client.sendRequest<SourceChange>(
            'rust-analyzer/ssr',
            ssrRequest,
        );

        await applySourceChange(ctx, change);
    };
}

interface SsrRequest {
    arg: string;
}
