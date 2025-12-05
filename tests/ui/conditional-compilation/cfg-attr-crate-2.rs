// https://github.com/rust-lang/rust/issues/21833#issuecomment-72353044

//@ compile-flags: --cfg broken --check-cfg=cfg(broken)

#![crate_type = "lib"]
#![cfg_attr(broken, no_core)] //~ ERROR the `#[no_core]` attribute is an experimental feature

pub struct S {}
