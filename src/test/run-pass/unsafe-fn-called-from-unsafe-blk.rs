// -*- rust -*-
//
// See also: compile-fail/unsafe-fn-called-from-safe.rs

unsafe fn f() { ret; }

fn g() {
    unsafe {
        f();
    }
}

fn h() unsafe {
    f();
}

fn main() {
}
