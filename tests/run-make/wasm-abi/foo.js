const fs = require('fs');
const process = require('process');
const assert = require('assert');
const buffer = fs.readFileSync(process.argv[2]);

const m = new WebAssembly.Module(buffer);
const i = new WebAssembly.Instance(m, {
  host: {
    two_i32: () => [100, 101],
    two_i64: () => [102n, 103n],
    two_f32: () => [104, 105],
    two_f64: () => [106, 107],
    mishmash: () => [108, 109, 110, 111n, 112, 113],
  }
});

assert.deepEqual(i.exports.return_two_i32(), [1, 2])
assert.deepEqual(i.exports.return_two_i64(), [3, 4])
assert.deepEqual(i.exports.return_two_f32(), [5, 6])
assert.deepEqual(i.exports.return_two_f64(), [7, 8])
assert.deepEqual(i.exports.return_mishmash(), [9, 10, 11, 12, 13, 14])
i.exports.call_imports();
