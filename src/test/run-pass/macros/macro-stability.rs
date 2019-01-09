// run-pass
// aux-build:unstable-macros.rs

#![feature(unstable_macros)]

#[macro_use] extern crate unstable_macros;

#[unstable(feature = "local_unstable", issue = "0")]
macro_rules! local_unstable { () => () }

fn main() {
    unstable_macro!();
    local_unstable!();
}
