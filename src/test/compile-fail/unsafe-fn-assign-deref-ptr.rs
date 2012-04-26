// -*- rust -*-

fn f(p: *u8) {
    *p = 0u8; //! ERROR dereference of unsafe pointer requires unsafe function or block
    ret;
}

fn main() {
}
