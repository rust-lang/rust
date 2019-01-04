// compile-flags: --cfg broken

#![crate_type = "lib"]
#![cfg_attr(broken, no_std, no_core)] //~ ERROR no_core is experimental

pub struct S {}
