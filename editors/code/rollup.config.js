// @ts-check

import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import nodeBuiltins from 'builtin-modules';

/** @type { import('rollup').RollupOptions } */
export default {
    input: 'out/src/main.js',
    plugins: [
        resolve({
            preferBuiltins: true
        }),
        commonjs()
    ],
    external: [...nodeBuiltins, 'vscode'],
    output: {
        file: './out/src/main.js',
        format: 'cjs'
    }
};
