//@ compile-flags: --cfg broken --check-cfg=cfg(broken)

#![crate_type = "lib"]
#![cfg_attr(broken, no_std, no_core)]
//~^ ERROR the `#[no_core]` attribute is an experimental feature

pub struct S {}
