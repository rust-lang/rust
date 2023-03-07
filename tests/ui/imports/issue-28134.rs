// compile-flags: --test

#![allow(soft_unstable)]
#![test] //~ ERROR cannot determine resolution for the attribute macro `test`
//~^ ERROR 4:1: 4:9: `test` attribute cannot be used at crate level
