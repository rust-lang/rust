//@ aux-build:impl-const.rs
//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

extern crate impl_const;

use impl_const::*;

pub fn main() {
    let n = Num::<5>;
    n.five();
}
