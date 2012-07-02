// -*- rust -*-

fn f(p: *u8) -> u8 {
    ret *p; //~ ERROR dereference of unsafe pointer requires unsafe function or block
}

fn main() {
}
