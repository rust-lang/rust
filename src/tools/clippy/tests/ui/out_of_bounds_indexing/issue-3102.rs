#![warn(clippy::out_of_bounds_indexing)]
#![allow(clippy::no_effect, const_err)]

fn main() {
    let x = [1, 2, 3, 4];

    // issue 3102
    let num = 1;
    &x[num..10]; // should trigger out of bounds error
    &x[10..num]; // should trigger out of bounds error
}
