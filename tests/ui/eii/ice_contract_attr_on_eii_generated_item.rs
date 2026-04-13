//@ compile-flags: --crate-type rlib

#![feature(extern_item_impls)]
#![feature(contracts)]
#![allow(incomplete_features)]

#[eii]
#[core::contracts::ensures]
//~^ ERROR contract annotations is only supported in functions with bodies
//~| ERROR contract annotations can only be used on functions
fn implementation();
