//@ check-pass
#![feature(const_block_items)]

const { assert!(true) }
const { assert!(2 + 2 == 4) }

fn main() {}
