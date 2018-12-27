//
// compile-flags: --cfg broken

// https://github.com/rust-lang/rust/issues/21833#issuecomment-72353044

#![cfg_attr(broken, no_core)] //~ ERROR no_core is experimental

fn main() { }
