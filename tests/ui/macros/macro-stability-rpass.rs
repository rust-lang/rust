//@ run-pass
//@ aux-build:unstable-macros.rs

#![unstable(feature = "one_two_three_testing", issue = "none")]
#![feature(staged_api, unstable_macros, local_unstable)]

#[macro_use] extern crate unstable_macros;

#[unstable(feature = "local_unstable", issue = "none")]
macro_rules! local_unstable { () => () }

fn main() {
    unstable_macro!();
    local_unstable!();
}
