#![feature(inclusive_range_syntax, plugin)]
#![plugin(clippy)]

#![deny(indexing_slicing)]
#![deny(out_of_bounds_indexing)]
#![allow(no_effect, unnecessary_operation)]

fn main() {
    let x = [1,2,3,4];
    x[0];
    x[3];
    x[4]; //~ERROR: const index is out of bounds
    x[1 << 3]; //~ERROR: const index is out of bounds
    &x[1..5]; //~ERROR: range is out of bounds
    &x[0..3];
    &x[0...4]; //~ERROR: range is out of bounds
    &x[..];
    &x[1..];
    &x[4..];
    &x[5..]; //~ERROR: range is out of bounds
    &x[..4];
    &x[..5]; //~ERROR: range is out of bounds

    let y = &x;
    y[0]; //~ERROR: indexing may panic
    &y[1..2]; //~ERROR: slicing may panic
    &y[..];
    &y[0...4]; //~ERROR: slicing may panic

    let empty: [i8; 0] = [];
    empty[0]; //~ERROR: const index is out of bounds
    &empty[1..5]; //~ERROR: range is out of bounds
    &empty[0...4]; //~ERROR: range is out of bounds
    &empty[..];
    &empty[0..];
    &empty[0..0];
    &empty[0...0]; //~ERROR: range is out of bounds
    &empty[..0];
    &empty[1..]; //~ERROR: range is out of bounds
    &empty[..4]; //~ERROR: range is out of bounds
}
