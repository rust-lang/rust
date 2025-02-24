//@ edition: 2018
#![feature(cfg_accessible)]
#![cfg(accessible(::std::boxed::Box))] //~ ERROR: `cfg(accessible(..))` cannot be used as crate attribute

fn main() {}
