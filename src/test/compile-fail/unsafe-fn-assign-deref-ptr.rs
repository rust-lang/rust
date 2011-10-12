// -*- rust -*-
// error-pattern: unsafe operation requires unsafe function or block

fn f(p: *u8) {
    *p = 0u8;
    ret;
}

fn main() {
    f();
}
