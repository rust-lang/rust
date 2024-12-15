// Ensure that `rustc-cfg-placeholder` isn't visible to proc-macros.
//@proc-macro: cfg-placeholder.rs
//@check-pass
#![feature(cfg_eval)]
#[macro_use] extern crate cfg_placeholder;

#[cfg_eval]
#[my_proc_macro]
#[cfg_attr(FALSE, my_attr1)]
#[cfg_attr(all(), my_attr2)]
struct S {}

fn main() {}
