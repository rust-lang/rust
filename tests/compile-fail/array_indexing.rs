#![feature(inclusive_range_syntax, plugin)]
#![plugin(clippy)]

#![deny(indexing_slicing)]
#![deny(out_of_bounds_indexing)]
#![allow(no_effect)]

fn main() {
    let x = [1,2,3,4];
    x[0];
    x[3];
    x[4]; //~ERROR: indexing may panic
          //~^ ERROR: const index is out of bounds
    x[1 << 3]; //~ERROR: indexing may panic
               //~^ ERROR: const index is out of bounds
    &x[1..5]; //~ERROR: slicing may panic
              //~^ ERROR: range is out of bounds
    &x[0..3];
    &x[0...4]; //~ERROR: slicing may panic
               //~^ ERROR: range is out of bounds
    &x[..];
    &x[1..];
    &x[..4];
    &x[..5]; //~ERROR: slicing may panic
             //~^ ERROR: range is out of bounds

    let y = &x;
    y[0]; //~ERROR: indexing may panic
    &y[1..2]; //~ERROR: slicing may panic
    &y[..];
    &y[0...4]; //~ERROR: slicing may panic
}
