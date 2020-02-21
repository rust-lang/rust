import * as vscode from 'vscode';
import { ensureServerBinary } from '../installation/server';
import { Ctx, Cmd } from '../ctx';
import { spawnSync } from 'child_process';

export function serverVersion(ctx: Ctx): Cmd {
    return async () => {
        const binaryPath = await ensureServerBinary(ctx.config.serverSource);

        if (binaryPath == null) {
            throw new Error(
                "Rust Analyzer Language Server is not available. " +
                "Please, ensure its [proper installation](https://rust-analyzer.github.io/manual.html#installation)."
            );
        }

        const res = spawnSync(binaryPath, ["--version"]);
        const version = res.output?.filter(x => x !== null).map(String).join(" ");
        vscode.window.showInformationMessage('rust-analyzer version : ' + version);
    };
}

