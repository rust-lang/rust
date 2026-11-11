//@ aux-crate:priv:equivalent=equivalent.rs
//@ compile-flags: -Zunstable-options --extern equivalent
//@ edition:2018

extern crate equivalent;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayoutError;
