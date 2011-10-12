// -*- rust -*-
//
// See also: compile-fail/unsafe-fn-called-from-safe.rs

unsafe fn f() { ret; }

unsafe fn g() {
    f();
}

fn main() {
    ret;
}
