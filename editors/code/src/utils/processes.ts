'use strict';

import * as cp from 'child_process';
import ChildProcess = cp.ChildProcess;

import { join } from 'path';

const isWindows = process.platform === 'win32';
const isMacintosh = process.platform === 'darwin';
const isLinux = process.platform === 'linux';

// this is very complex, but is basically copy-pased from VSCode implementation here:
// https://github.com/Microsoft/vscode-languageserver-node/blob/dbfd37e35953ad0ee14c4eeced8cfbc41697b47e/client/src/utils/processes.ts#L15

// And see discussion at
// https://github.com/rust-analyzer/rust-analyzer/pull/1079#issuecomment-478908109

export function terminate(process: ChildProcess, cwd?: string): boolean {
    if (isWindows) {
        try {
            // This we run in Atom execFileSync is available.
            // Ignore stderr since this is otherwise piped to parent.stderr
            // which might be already closed.
            const options: any = {
                stdio: ['pipe', 'pipe', 'ignore']
            };
            if (cwd) {
                options.cwd = cwd;
            }
            cp.execFileSync(
                'taskkill',
                ['/T', '/F', '/PID', process.pid.toString()],
                options
            );
            return true;
        } catch (err) {
            return false;
        }
    } else if (isLinux || isMacintosh) {
        try {
            const cmd = join(__dirname, 'terminateProcess.sh');
            const result = cp.spawnSync(cmd, [process.pid.toString()]);
            return result.error ? false : true;
        } catch (err) {
            return false;
        }
    } else {
        process.kill('SIGKILL');
        return true;
    }
}
