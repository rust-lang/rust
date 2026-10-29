//@ run-rustfix

#![allow(unused)]

// Regression test for https://github.com/rust-lang/rust/issues/157455.
mod public {
    pub struct Hi;
}

mod testing {
    use super::public::Hi;
}

use testing::Hi;
//~^ ERROR struct import `Hi` is private [E0603]

fn main() {}
