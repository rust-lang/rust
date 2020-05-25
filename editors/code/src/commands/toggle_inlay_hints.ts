import * as vscode from 'vscode';
import { Ctx, Cmd } from '../ctx';

export function toggleInlayHints(ctx: Ctx): Cmd {
    return async () => {
        await vscode
            .workspace
            .getConfiguration(`${ctx.config.rootSection}.inlayHints`)
            .update('enable', !ctx.config.inlayHints.enable, vscode.ConfigurationTarget.Workspace);
    };
}
