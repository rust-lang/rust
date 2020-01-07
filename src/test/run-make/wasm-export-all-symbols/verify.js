const fs = require('fs');
const process = require('process');
const assert = require('assert');
const buffer = fs.readFileSync(process.argv[2]);

let m = new WebAssembly.Module(buffer);
let list = WebAssembly.Module.exports(m);
console.log('exports', list);

const my_exports = {};
let nexports_fn = 0;
let nexports_global = 0;
for (const entry of list) {
  if (entry.kind == 'function'){
    nexports_fn += 1;
  }
  if (entry.kind == 'global'){
    nexports_global += 1;
  }
  my_exports[entry.name] = true;
}

if (my_exports.foo === undefined)
  throw new Error("`foo` wasn't defined");

if (my_exports.FOO === undefined)
  throw new Error("`FOO` wasn't defined");

if (my_exports.main === undefined) {
  if (nexports_fn != 1)
    throw new Error("should only have one function export");
} else {
  if (nexports_fn != 2)
    throw new Error("should only have two function exports");
}

if (nexports_global != 1)
  throw new Error("should only have one static export");
