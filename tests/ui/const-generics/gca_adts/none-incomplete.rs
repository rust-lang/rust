//! This test file ought to be part of `none.rs`, but, type-relative paths are difficult

#![feature(gca_adts, adt_const_params)]

use std::gca;
use std::marker::ConstParamTy;

#[derive(ConstParamTy, PartialEq, Eq)]
enum MyOption {
    MySome,
    MyNone,
}

use MyOption::*;

struct Struct<const A: MyOption>;

fn main() {
    let _: Struct<gca!(<MyOption>::MyNone)>;
    //~^ ERROR complex const arguments must be placed inside of a `const` block
}
