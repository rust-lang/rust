// -*- rust -*-
// error-pattern: unsafe functions can only be called

unsafe fn f() { ret; }

fn main() {
    let x = f;
    x();
}
