//@check-pass

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
    let _: Struct<gca!(MyNone)>;
    let _: Struct<gca!(MyOption::MyNone)>;
}
