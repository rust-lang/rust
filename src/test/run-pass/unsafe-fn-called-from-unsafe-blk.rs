#![allow(dead_code)]
//
// See also: compile-fail/unsafe-fn-called-from-safe.rs

// pretty-expanded FIXME #23616

unsafe fn f() { return; }

fn g() {
    unsafe {
        f();
    }
}

pub fn main() {
}
