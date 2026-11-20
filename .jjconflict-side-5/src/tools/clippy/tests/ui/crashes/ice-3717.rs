#![deny(clippy::implicit_hasher)]

use std::collections::HashSet;

fn main() {}

pub fn ice_3717(_: &HashSet<usize>) {
    //~^ implicit_hasher

    let _ = [0u8; 0];
    let _: HashSet<usize> = HashSet::new();
}
