// -*- rust -*-
//
// See also: compile-fail/unsafe-fn-called-from-safe.rs

unsafe fn f() { ret; }

fn main() {
    unsafe {
        f();
    }
}
