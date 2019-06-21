// aux-build:unstable-macros.rs

#![feature(staged_api)]
#[macro_use] extern crate unstable_macros;

#[unstable(feature = "local_unstable", issue = "0")]
macro_rules! local_unstable { () => () }

fn main() {
    local_unstable!(); //~ ERROR use of unstable library feature 'local_unstable'
    unstable_macro!(); //~ ERROR use of unstable library feature 'unstable_macros'
}
