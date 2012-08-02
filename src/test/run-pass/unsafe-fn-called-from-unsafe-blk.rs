// -*- rust -*-
//
// See also: compile-fail/unsafe-fn-called-from-safe.rs

unsafe fn f() { return; }

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
