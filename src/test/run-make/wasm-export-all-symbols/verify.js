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
let list = WebAssembly.Module.exports(m);
console.log('exports', list);

const my_exports = {};
let nexports = 0;
for (const entry of list) {
  if (entry.kind !== 'function')
    continue;
  my_exports[entry.name] = true;
  nexports += 1;
}

if (nexports != 1)
  throw new Error("should only have one function export");
if (my_exports.foo === undefined)
  throw new Error("`foo` wasn't defined");
