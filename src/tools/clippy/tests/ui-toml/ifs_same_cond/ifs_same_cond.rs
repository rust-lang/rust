#![warn(clippy::ifs_same_cond)]
#![allow(clippy::if_same_then_else, clippy::comparison_chain, clippy::needless_else)]

fn main() {}

fn issue10272() {
    use std::cell::Cell;

    // Because the `ignore-interior-mutability` configuration
    // is set to ignore for `std::cell::Cell`, the following `get()` calls
    // should trigger warning
    let x = Cell::new(true);
    if x.get() {
        //~^ ifs_same_cond
    } else if !x.take() {
    } else if x.get() {
    } else {
    }
}
