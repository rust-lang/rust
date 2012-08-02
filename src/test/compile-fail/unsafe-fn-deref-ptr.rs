// -*- rust -*-

fn f(p: *u8) -> u8 {
    return *p; //~ ERROR dereference of unsafe pointer requires unsafe function or block
}

fn main() {
}
