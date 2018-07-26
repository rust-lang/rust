// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
