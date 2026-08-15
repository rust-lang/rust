// issue: <https://github.com/rust-lang/rust/issues/11508>
// Test pattern matching on a tuple struct defined in an external crate.
//@ run-pass
//@ aux-build:tuple-struct-cross-crate-aux.rs

extern crate tuple_struct_cross_crate_aux as rand;

use rand::{Closed01, random};

fn main() {
    let Closed01(val) = random::<Closed01<f32>>();
    println!("{}", val);
}
