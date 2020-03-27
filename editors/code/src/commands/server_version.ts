import * as vscode from "vscode";
import { spawnSync } from "child_process";
import { Ctx, Cmd } from '../ctx';

export function serverVersion(ctx: Ctx): Cmd {
    return async () => {
        const { stdout } = spawnSync(ctx.serverPath, ["--version"], { encoding: "utf8" });
        const commitHash = stdout.slice(`rust-analyzer `.length).trim();
        const { releaseTag } = ctx.config.package;

        void vscode.window.showInformationMessage(
            `rust-analyzer version: ${releaseTag ?? "unreleased"} (${commitHash})`
        );
    };
}
