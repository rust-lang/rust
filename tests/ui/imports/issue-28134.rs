//@ compile-flags: --test

#![allow(soft_unstable)]
#![test]
//~^ ERROR 4:1: 4:9: `test` attribute cannot be used at crate level
