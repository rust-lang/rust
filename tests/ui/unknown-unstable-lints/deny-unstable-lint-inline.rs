//@ check-fail

#![deny(unknown_lints)]
#![allow(test_unstable_lint)]
//~^ ERROR unknown lint: `test_unstable_lint`

fn main() {}
