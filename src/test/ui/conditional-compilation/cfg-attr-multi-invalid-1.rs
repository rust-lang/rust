// compile-flags: --cfg broken

#![crate_type = "lib"]
#![cfg_attr(broken, no_core, no_std)] //~ ERROR no_core is experimental

pub struct S {}
