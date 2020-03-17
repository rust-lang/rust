import * as vscode from "vscode";
import { spawnSync } from "child_process";
import { Ctx, Cmd } from '../ctx';

export function serverVersion(ctx: Ctx): Cmd {
    return async () => {
        const version = spawnSync(ctx.serverPath, ["--version"], { encoding: "utf8" }).stdout;
        vscode.window.showInformationMessage('rust-analyzer version : ' + version);
    };
}
