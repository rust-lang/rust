// -*- rust -*-
// error-pattern: safe function calls function marked unsafe
// xfail-test

unsafe fn f() { ret; }

fn main() {
    f();
}
