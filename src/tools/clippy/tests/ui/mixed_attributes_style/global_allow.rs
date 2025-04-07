// https://github.com/rust-lang/rust-clippy/issues/12436
//@check-pass
#![allow(clippy::mixed_attributes_style)]

#[path = "auxiliary/submodule.rs"]
mod submodule;

fn main() {}
