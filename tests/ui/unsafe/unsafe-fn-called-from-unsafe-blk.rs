//@ run-pass

#![allow(dead_code)]
//
// See also: ui/unsafe/unsafe-fn-called-from-safe.rs


unsafe fn f() { return; }

fn g() {
    unsafe {
        f();
    }
}

pub fn main() {
}
