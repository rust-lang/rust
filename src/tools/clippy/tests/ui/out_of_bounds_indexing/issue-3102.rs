#![warn(clippy::out_of_bounds_indexing)]
#![allow(clippy::no_effect)]

fn main() {
    let x = [1, 2, 3, 4];

    // issue 3102
    let num = 1;
    &x[num..10];
    //~^ out_of_bounds_indexing

    &x[10..num];
    //~^ out_of_bounds_indexing
}
