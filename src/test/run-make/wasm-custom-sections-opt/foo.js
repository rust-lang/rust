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

sections = WebAssembly.Module.customSections(m, "foo");
console.log('section foo', sections);
assert.strictEqual(sections.length, 1, "didn't create `foo` section");
section = new Uint8Array(sections[0]);
console.log('contents', section);
assert.strictEqual(section.length, 4, "didn't concatenate `foo` sections");

process.exit(0);
