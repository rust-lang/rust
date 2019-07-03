// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Issue 33903:
// Built-in indexing should be used even when the index is not
// trivially an integer
// Only built-in indexing can be used in constant expressions

const FOO: i32 = [12, 34][0 + 1];

fn main() {}
