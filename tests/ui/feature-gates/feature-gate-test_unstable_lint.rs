//@ check-pass

// `test_unstable_lint` is for testing and should never be stabilized.
#![allow(test_unstable_lint)]
//~^ WARNING unknown lint: `test_unstable_lint`

fn main() {}
