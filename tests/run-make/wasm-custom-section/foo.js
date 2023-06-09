const fs = require('fs');
const process = require('process');
const assert = require('assert');
const buffer = fs.readFileSync(process.argv[2]);

let m = new WebAssembly.Module(buffer);
let sections = WebAssembly.Module.customSections(m, "baz");
console.log('section baz', sections);
assert.strictEqual(sections.length, 1);
let section = new Uint8Array(sections[0]);
console.log('contents', section);
assert.strictEqual(section.length, 2);
assert.strictEqual(section[0], 7);
assert.strictEqual(section[1], 8);

sections = WebAssembly.Module.customSections(m, "bar");
console.log('section bar', sections);
assert.strictEqual(sections.length, 1, "didn't pick up `bar` section from dependency");
section = new Uint8Array(sections[0]);
console.log('contents', section);
assert.strictEqual(section.length, 2);
assert.strictEqual(section[0], 3);
assert.strictEqual(section[1], 4);

sections = WebAssembly.Module.customSections(m, "foo");
console.log('section foo', sections);
assert.strictEqual(sections.length, 1, "didn't create `foo` section");
section = new Uint8Array(sections[0]);
console.log('contents', section);
assert.strictEqual(section.length, 4, "didn't concatenate `foo` sections");
assert.strictEqual(section[0], 5);
assert.strictEqual(section[1], 6);
assert.strictEqual(section[2], 1);
assert.strictEqual(section[3], 2);

process.exit(0);
