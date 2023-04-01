// run-rustfix
#![warn(clippy::wildcard_imports)]
#![allow(unused, clippy::unnecessary_wraps, clippy::let_unit_value)]

// Test for #10580, the lint should **not** ignore it.

fn foofoo() {}

mod outer {
    mod inner {
        use super::super::*;
        fn barbar() {
            let _ = foofoo();
        }
    }
}

fn main() {}
