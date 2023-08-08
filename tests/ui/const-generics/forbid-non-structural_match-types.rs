#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct A;

struct B<const X: A>; // ok

struct C;

struct D<const X: C>; //~ ERROR `C` must implement `ConstParamTy` to be used as the type of a const generic parameter

fn main() {}
