#![deny(clippy::implicit_hasher)]

//@no-rustfix: need to change the suggestion to a multipart suggestion

use std::collections::HashSet;

fn main() {}

pub fn ice_3717(_: &HashSet<usize>) {
    //~^ ERROR: parameter of type `HashSet` should be generalized over different hashers
    let _ = [0u8; 0];
    let _: HashSet<usize> = HashSet::new();
}
