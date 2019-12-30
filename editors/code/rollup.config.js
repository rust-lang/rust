import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import nodeBuiltins from 'builtin-modules';

export default {
    input: 'src/main.ts',
    plugins: [
        typescript(),
        resolve({
            preferBuiltins: true
        }),
        commonjs({
            namedExports: {
                // squelch missing import warnings
                'vscode-languageclient': ['CreateFile', 'RenameFile', 'ErrorCodes']
            }
        })
    ],
    external: [...nodeBuiltins, 'vscode'],
    output: {
        file: './out/main.js',
        format: 'cjs'
    }
};
