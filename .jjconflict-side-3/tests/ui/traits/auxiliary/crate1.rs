//@ aux-crate:crate2=crate2.rs

pub trait Trait {}

pub fn foo(_arg: impl Trait) {}

pub fn bar(_arg: impl crate2::Trait) {}
