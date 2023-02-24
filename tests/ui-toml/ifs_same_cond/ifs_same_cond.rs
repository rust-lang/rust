#![warn(clippy::ifs_same_cond)]
#![allow(clippy::if_same_then_else, clippy::comparison_chain)]

fn main() {}

fn issue10272() {
    use std::cell::Cell;

    let x = Cell::new(true);
    if x.get() {
    } else if !x.take() {
    } else if x.get() {
        // ok, x is interior mutable type
    } else {
    }

    let a = [Cell::new(true)];
    if a[0].get() {
    } else if a[0].take() {
    } else if a[0].get() {
        // ok, a contains interior mutable type
    } else {
    }
}
