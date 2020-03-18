// @ts-check

import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import nodeBuiltins from 'builtin-modules';

/** @type { import('rollup').RollupOptions } */
export default {
    input: 'out/main.js',
    plugins: [
        resolve({
            preferBuiltins: true
        }),
        commonjs({
            namedExports: {
                // squelch missing import warnings
                'vscode-languageclient': ['CreateFile', 'RenameFile', 'ErrorCodes', 'WorkDoneProgress', 'WorkDoneProgressBegin', 'WorkDoneProgressReport', 'WorkDoneProgressEnd']
            }
        })
    ],
    external: [...nodeBuiltins, 'vscode'],
    output: {
        file: './out/main.js',
        format: 'cjs',
        exports: 'named'
    }
};
