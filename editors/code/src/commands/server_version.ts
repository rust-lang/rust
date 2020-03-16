import * as vscode from 'vscode';
import { ensureServerBinary } from '../installation/server';
import { Ctx, Cmd } from '../ctx';
import { spawnSync } from 'child_process';

export function serverVersion(ctx: Ctx): Cmd {
    return async () => {
        const binaryPath = await ensureServerBinary(ctx.config, ctx.state);

        if (binaryPath == null) {
            throw new Error(
                "Rust Analyzer Language Server is not available. " +
                "Please, ensure its [proper installation](https://rust-analyzer.github.io/manual.html#installation)."
            );
        }

        const version = spawnSync(binaryPath, ["--version"], { encoding: "utf8" }).stdout;
        vscode.window.showInformationMessage('rust-analyzer version : ' + version);
    };
}
