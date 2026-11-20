//@ build-pass

//! Regression test #124478.

use std::mem::offset_of;

struct S {
    v: u8,
    w: u16,
}

impl S {
    fn return_static_slice() -> &'static [usize] {
        &[offset_of!(Self, v), offset_of!(Self, w)]
    }
    fn use_reference() -> usize {
        let r = &offset_of!(Self, v);
        *r * 6
    }
}

fn main() {
}
