// compile-flags: --cfg broken

#![feature(cfg_attr_multi)]
#![crate_type = "lib"]
#![cfg_attr(broken, no_core, no_std)] //~ ERROR no_core is experimental

pub struct S {}
