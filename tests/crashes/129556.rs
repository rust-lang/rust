//@ known-bug: rust-lang/rust#129556

#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

use core::marker::ConstParamTy;

// #[derive(ConstParamTy, PartialEq, Eq)]
// struct StructOrEnum;

#[derive(ConstParamTy, PartialEq, Eq)]
enum StructOrEnum {
    A,
}

trait TraitParent<const SMTH: StructOrEnum = { StructOrEnum::A }> {}

trait TraitChild<const SMTH: StructOrEnum = { StructOrEnum::A }>: TraitParent<SMTH> {}

impl TraitParent for StructOrEnum {}

// ICE occurs
impl TraitChild for StructOrEnum {}

// ICE does not occur
// impl TraitChild<{ StructOrEnum::A }> for StructOrEnum {}
