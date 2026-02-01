// Regression test for #129556: trait const param default used to ICE in ArgFolder.

#![allow(incomplete_features)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

use core::marker::ConstParamTy;
//~^ ERROR failed to resolve: you might be missing crate `core`

#[derive(ConstParamTy, PartialEq, Eq)]
enum StructOrEnum {
    A,
}

trait TraitParent<const SMTH: StructOrEnum = { StructOrEnum::A }> {}
//~^ ERROR `StructOrEnum` must implement `ConstParamTy`

trait TraitChild<const SMTH: StructOrEnum = { StructOrEnum::A }>: TraitParent<SMTH> {}
//~^ ERROR `StructOrEnum` must implement `ConstParamTy`

impl TraitParent for StructOrEnum {}

impl TraitChild for StructOrEnum {}
//~^ ERROR mismatched types

fn main() {}
