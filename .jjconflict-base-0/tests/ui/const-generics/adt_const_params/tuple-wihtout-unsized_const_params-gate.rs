//! Ensure we allow tuples behind `min_adt_const_params`
//@check-pass
#![feature(min_adt_const_params)]
#![allow(dead_code)]

use std::marker::ConstParamTy;

fn foo<const N: (i32, u32, i16)>() {}
fn foo2<const TUP: Something>() {}

#[derive(PartialEq, Eq, ConstParamTy)]
struct Something(i8, i16, i32);

fn main() {}
