// https://github.com/rust-lang/rust/issues/21833#issuecomment-72353044

// compile-flags: --cfg broken

#![crate_type = "lib"]
#![cfg_attr(broken, no_core)] //~ ERROR no_core is experimental

pub struct S {}
