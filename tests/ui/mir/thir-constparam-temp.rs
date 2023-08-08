// build-pass

#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct Yikes;

impl Yikes {
    fn mut_self(&mut self) {}
}

fn foo<const YIKES: Yikes>() {
    YIKES.mut_self()
    //~^ WARNING taking a mutable reference
}

fn main() {
    foo::<{ Yikes }>()
}
