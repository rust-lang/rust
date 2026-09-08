//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no
#![feature(rustc_attrs)]

#[rustc_dyn_incompatible_trait]
pub trait DynIncompatible {}

pub fn f(_x: &dyn DynIncompatible) {}
