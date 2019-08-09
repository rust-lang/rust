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

if (my_exports.foo === undefined)
  throw new Error("`foo` wasn't defined");

if (my_exports.main === undefined) {
  if (nexports != 1)
    throw new Error("should only have one function export");
} else {
  if (nexports != 2)
    throw new Error("should only have two function exports");
}
