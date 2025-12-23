//@ check-fail

#![feature(const_block_items)]

const { assert!(false) }
//~^ ERROR: evaluation panicked: assertion failed: false [E0080]
const { assert!(2 + 2 == 5) }
//~^ ERROR: evaluation panicked: assertion failed: 2 + 2 == 5 [E0080]

fn main() {}
