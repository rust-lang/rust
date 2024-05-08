//@ known-bug: rust-lang/rust#124436
//@ compile-flags: -Zdump-mir=all -Zpolymorphize=on

pub trait TraitCat {}
pub trait TraitDog {}

pub fn gamma<T: TraitCat + TraitDog>(t: [TraitDog; 32]) {}
