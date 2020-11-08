// run-pass
// pretty-expanded FIXME #23616

// Tests that type parameters with the `Copy` are implicitly copyable.

#![allow(dead_code)]

fn can_copy_copy<T: Copy>(v: T) {
    let _a = v;
    let _b = v;
}

pub fn main() {}
