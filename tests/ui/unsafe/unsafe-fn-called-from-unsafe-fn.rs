//@ run-pass

#![allow(dead_code)]
//
// See also: ui/unsafe/unsafe-fn-called-from-safe.rs


unsafe fn f() { return; }

unsafe fn g() {
    f();
}

pub fn main() {
    return;
}
