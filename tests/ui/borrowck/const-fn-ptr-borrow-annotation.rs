// Regression test for https://github.com/rust-lang/rust/issues/160255.

use std::mem;

const A: fn() = unsafe {
    mem::transmute({
        fn fun() {}
        let _ = fun as fn();
        {
            let s = [0; 10];
            &s //~ ERROR: `s` does not live long enough [E0597]
        }
    })
};

fn main() {}
