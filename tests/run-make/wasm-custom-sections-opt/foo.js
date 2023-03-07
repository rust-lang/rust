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
