import * as vscode from 'vscode';
import * as ra from "../rust-analyzer-api";

import { Ctx, Cmd } from '../ctx';
import { applySourceChange } from '../source_change';

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

        const change = await client.sendRequest(ra.ssr, { arg: request });

        await applySourceChange(ctx, change);
    };
}
