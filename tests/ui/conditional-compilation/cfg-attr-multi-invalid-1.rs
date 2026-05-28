//@ compile-flags: --cfg broken --check-cfg=cfg(broken)

#![crate_type = "lib"]
#![cfg_attr(broken, no_core, no_std)]
//~^ ERROR the `#[no_core]` attribute is an experimental feature

pub struct S {}
