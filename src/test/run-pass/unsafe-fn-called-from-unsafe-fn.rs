// -*- rust -*-
//
// See also: compile-fail/unsafe-fn-called-from-safe.rs

unsafe fn f() { return; }

unsafe fn g() {
    f();
}

fn main() {
    return;
}
