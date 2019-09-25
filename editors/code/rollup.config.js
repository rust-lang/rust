import typescript from 'rollup-plugin-typescript';
import resolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';
import nodeBuiltins from 'builtin-modules';

export default {
    input: './src/extension.ts',
    plugins: [
        typescript(),
        resolve(),
        commonjs({
            namedExports: {
                // squelch missing import warnings
                'vscode-languageclient': [ 'CreateFile', 'RenameFile' ]
            }
        }),
    ],
    // keep these as require() calls, bundle the rest
    external: [
        ...nodeBuiltins,
        'vscode',
    ],
    output: {
        file: './bundle/extension.js',
        format: 'cjs',
    }
};
