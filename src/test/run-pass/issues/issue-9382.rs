// pretty-expanded FIXME #23616


// run-pass
#![allow(dead_code)]
#![feature(box_syntax)]

// Tests for a previous bug that occurred due to an interaction
// between struct field initialization and the auto-coercion
// from a vector to a slice. The drop glue was being invoked on
// the temporary slice with a wrong type, triggering an LLVM assert.


struct Thing1<'a> {
    baz: &'a [Box<isize>],
    bar: Box<u64>,
}

struct Thing2<'a> {
    baz: &'a [Box<isize>],
    bar: u64,
}

pub fn main() {
    let _t1_fixed = Thing1 {
        baz: &[],
        bar: box 32,
    };
    Thing1 {
        baz: &Vec::new(),
        bar: box 32,
    };
    let _t2_fixed = Thing2 {
        baz: &[],
        bar: 32,
    };
    Thing2 {
        baz: &Vec::new(),
        bar: 32,
    };
}
