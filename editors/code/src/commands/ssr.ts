import * as vscode from 'vscode';
import * as ra from "../rust-analyzer-api";

import { Ctx, Cmd } from '../ctx';

export function ssr(ctx: Ctx): Cmd {
    return async () => {
        const client = ctx.client;
        if (!client) return;

        const options: vscode.InputBoxOptions = {
            value: "() ==>> ()",
            prompt: "Enter request, for example 'Foo($a:expr) ==> Foo::new($a)' ",
            validateInput: async (x: string) => {
                try {
                    await client.sendRequest(ra.ssr, { query: x, parseOnly: true });
                } catch (e) {
                    return e.toString();
                }
                return null;
            }
        };
        const request = await vscode.window.showInputBox(options);
        if (!request) return;

        const edit = await client.sendRequest(ra.ssr, { query: request, parseOnly: false });

        await vscode.workspace.applyEdit(client.protocol2CodeConverter.asWorkspaceEdit(edit));
    };
}
