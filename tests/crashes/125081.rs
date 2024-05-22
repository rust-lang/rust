//@ known-bug: rust-lang/rust#125081

use std::cell::Cell;

fn main() {
    let _: Cell<&str, "a"> = Cell::new('β);
}
