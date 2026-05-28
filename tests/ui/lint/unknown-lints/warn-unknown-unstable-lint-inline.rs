//@ check-pass

#![warn(unknown_lints)]
#![allow(test_unstable_lint)]
//~^ WARNING unknown lint: `test_unstable_lint`

fn main() {}
