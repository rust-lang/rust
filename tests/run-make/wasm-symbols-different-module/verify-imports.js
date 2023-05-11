const fs = require('fs');
const process = require('process');
const assert = require('assert');
const buffer = fs.readFileSync(process.argv[2]);

let m = new WebAssembly.Module(buffer);
let list = WebAssembly.Module.imports(m);
console.log('imports', list);
if (list.length !== process.argv.length - 3)
  throw new Error("wrong number of imports")

const imports = new Map();
for (let i = 3; i < process.argv.length; i++) {
  const [module, name] = process.argv[i].split('/');
  if (!imports.has(module))
    imports.set(module, new Map());
  imports.get(module).set(name, true);
}

for (let i of list) {
  if (imports.get(i.module) === undefined || imports.get(i.module).get(i.name) === undefined)
    throw new Error(`didn't find import of ${i.module}::${i.name}`);
  imports.get(i.module).delete(i.name);

  if (imports.get(i.module).size === 0)
    imports.delete(i.module);
}

console.log(imports);
if (imports.size !== 0) {
  throw new Error('extra imports');
}
