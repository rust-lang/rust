// compile-pass
// aux-build:dollar-crate.rs

extern crate dollar_crate;

type S = u8;

macro_rules! check { () => {
    dollar_crate::normalize! {
        type A = $crate::S;
    }
}}

check!();

fn main() {}
