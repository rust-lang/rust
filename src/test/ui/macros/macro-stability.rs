// aux-build:unstable-macros.rs

#![feature(staged_api)]
#[macro_use] extern crate unstable_macros;

#[unstable(feature = "local_unstable", issue = "0")]
macro_rules! local_unstable { () => () }

fn main() {
    local_unstable!();
    unstable_macro!(); //~ ERROR: macro unstable_macro! is unstable
}
