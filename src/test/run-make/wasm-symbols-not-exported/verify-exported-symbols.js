const fs = require('fs');
const process = require('process');
const assert = require('assert');
const buffer = fs.readFileSync(process.argv[2]);

let m = new WebAssembly.Module(buffer);
let list = WebAssembly.Module.exports(m);
console.log('exports', list);

let bad = false;
for (let i = 0; i < list.length; i++) {
  const e = list[i];
  if (e.name == "foo" || e.kind != "function")
    continue;

  console.log('unexpected exported symbol:', e.name);
  bad = true;
}

if (bad)
  process.exit(1);
