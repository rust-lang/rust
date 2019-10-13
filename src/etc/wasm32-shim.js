// This is a small "shim" program which is used when wasm32 unit tests are run
// in this repository. This program is intended to be run in node.js and will
// load a wasm module into memory, instantiate it with a set of imports, and
// then run it.
//
// There's a bunch of helper functions defined here in `imports.env`, but note
// that most of them aren't actually needed to execute most programs. Many of
// these are just intended for completeness or debugging. Hopefully over time
// nothing here is needed for completeness.

const fs = require('fs');
const process = require('process');
const buffer = fs.readFileSync(process.argv[2]);

Error.stackTraceLimit = 20;

let m = new WebAssembly.Module(buffer);
let instance = new WebAssembly.Instance(m, {});
try {
  instance.exports.main();
} catch (e) {
  console.error(e);
  process.exit(101);
}
