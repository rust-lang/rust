//@ check-pass

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn use_block_test(x: i32) -> i32 {
    let x = { let x = x + 1; x }.use;
    x
}

fn main() {}
