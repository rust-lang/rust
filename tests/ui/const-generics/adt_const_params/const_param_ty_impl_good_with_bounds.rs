// check-pass

#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq)]
struct Bar;
#[derive(PartialEq, Eq)]
struct Foo(Bar);

impl ConstParamTy for Bar
where
  Foo: ConstParamTy {}

// impl checks means that this impl is only valid if `Bar: ConstParamTy` whic
// is only valid if `Foo: ConstParamTy` holds
impl ConstParamTy for Foo {}

fn main() {}
