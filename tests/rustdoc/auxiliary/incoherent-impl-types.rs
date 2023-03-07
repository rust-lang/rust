#![feature(rustc_attrs)]

#[rustc_has_incoherent_inherent_impls]
pub trait FooTrait {}

#[rustc_has_incoherent_inherent_impls]
pub struct FooStruct;
