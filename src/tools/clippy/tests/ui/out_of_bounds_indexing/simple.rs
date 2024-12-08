#![warn(clippy::out_of_bounds_indexing)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

fn main() {
    let x = [1, 2, 3, 4];

    &x[..=4];
    //~^ ERROR: range is out of bounds
    //~| NOTE: `-D clippy::out-of-bounds-indexing` implied by `-D warnings`
    &x[1..5];
    //~^ ERROR: range is out of bounds
    &x[5..];
    //~^ ERROR: range is out of bounds
    &x[..5];
    //~^ ERROR: range is out of bounds
    &x[5..].iter().map(|x| 2 * x).collect::<Vec<i32>>();
    //~^ ERROR: range is out of bounds
    &x[0..=4];
    //~^ ERROR: range is out of bounds

    &x[4..]; // Ok, should not produce stderr.
    &x[..4]; // Ok, should not produce stderr.
    &x[..]; // Ok, should not produce stderr.
    &x[1..]; // Ok, should not produce stderr.
    &x[2..].iter().map(|x| 2 * x).collect::<Vec<i32>>(); // Ok, should not produce stderr.

    &x[0..].get(..3); // Ok, should not produce stderr.
    &x[0..3]; // Ok, should not produce stderr.
}
