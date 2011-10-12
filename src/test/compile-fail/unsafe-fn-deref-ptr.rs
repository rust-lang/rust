// -*- rust -*-
// error-pattern: unsafe operation requires unsafe function or block

fn f(p: *u8) -> u8 {
    ret *p;
}

fn main() {
    f();
}
