#![warn(clippy::indexing_slicing)]
// We also check the out_of_bounds_indexing lint here, because it lints similar things and
// we want to avoid false positives.
#![warn(clippy::out_of_bounds_indexing)]
#![allow(clippy::no_effect, clippy::unnecessary_operation, clippy::useless_vec)]

fn main() {
    let x = [1, 2, 3, 4];
    let index: usize = 1;
    let index_from: usize = 2;
    let index_to: usize = 3;
    &x[index..];
    //~^ ERROR: slicing may panic
    &x[..index];
    //~^ ERROR: slicing may panic
    &x[index_from..index_to];
    //~^ ERROR: slicing may panic
    &x[index_from..][..index_to];
    //~^ ERROR: slicing may panic
    //~| ERROR: slicing may panic
    &x[5..][..10];
    //~^ ERROR: slicing may panic
    //~| ERROR: range is out of bounds
    //~| NOTE: `-D clippy::out-of-bounds-indexing` implied by `-D warnings`
    &x[0..][..3];
    //~^ ERROR: slicing may panic
    &x[1..][..5];
    //~^ ERROR: slicing may panic

    &x[0..].get(..3); // Ok, should not produce stderr.
    &x[0..3]; // Ok, should not produce stderr.

    let y = &x;
    &y[1..2];
    &y[0..=4];
    //~^ ERROR: range is out of bounds
    &y[..=4];
    //~^ ERROR: range is out of bounds

    &y[..]; // Ok, should not produce stderr.

    let v = vec![0; 5];
    &v[10..100];
    //~^ ERROR: slicing may panic
    &x[10..][..100];
    //~^ ERROR: slicing may panic
    //~| ERROR: range is out of bounds
    &v[10..];
    //~^ ERROR: slicing may panic
    &v[..100];
    //~^ ERROR: slicing may panic

    &v[..]; // Ok, should not produce stderr.
}
