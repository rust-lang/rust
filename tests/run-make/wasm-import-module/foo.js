const fs = require('fs');
const process = require('process');
const assert = require('assert');
const buffer = fs.readFileSync(process.argv[2]);

let m = new WebAssembly.Module(buffer);
let imports = WebAssembly.Module.imports(m);
console.log('imports', imports);
assert.strictEqual(imports.length, 2);

assert.strictEqual(imports[0].kind, 'function');
assert.strictEqual(imports[1].kind, 'function');

let modules = [imports[0].module, imports[1].module];
modules.sort();

assert.strictEqual(modules[0], './dep');
assert.strictEqual(modules[1], './me');
