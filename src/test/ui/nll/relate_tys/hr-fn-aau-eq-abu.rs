// Test an interesting corner case that has previously been legal: a fn that takes
// two arguments that are references with the same lifetime is in fact
// equivalent to a fn that takes two references with distinct
// lifetimes. This is true because the two functions can call one
// another -- effectively, the single lifetime `'a` is just inferred
// to be the intersection of the two distinct lifetimes.
#![feature(nll)]

use std::cell::Cell;

fn make_cell_aa() -> Cell<for<'a> fn(&'a u32, &'a u32)> {
    panic!()
}

fn aa_eq_ab() {
    let a: Cell<for<'a, 'b> fn(&'a u32, &'b u32)> = make_cell_aa();
    //~^ ERROR mismatched types
    drop(a);
}

fn main() { }
