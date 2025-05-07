//@ known-bug: #133199
//@ aux-build: aux133199.rs

extern crate aux133199;

use aux133199::FixedBitSet;

fn main() {
    FixedBitSet::<7>::new();
    //~^ ERROR
}
