// check-pass
// Regression test of #77475, this used to be ICE.

#![feature(decl_macro)]

use crate as _;

pub macro ice(){}

fn main() {}
