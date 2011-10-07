// -*- rust -*-
// error-pattern: safe function calls function marked unsafe

unsafe fn f() { ret; }

fn main() {
    f();
}
