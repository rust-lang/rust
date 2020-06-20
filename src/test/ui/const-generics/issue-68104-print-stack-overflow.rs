// aux-build:impl-const.rs
// run-pass

#![feature(const_generics)]
#![allow(incomplete_features)]

extern crate impl_const;

use impl_const::*;

pub fn main() {
    let n = Num::<5>;
    n.five();
}
