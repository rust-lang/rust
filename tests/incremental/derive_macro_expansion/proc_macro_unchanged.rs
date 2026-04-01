// This test tests that derive proc macro execution is cached.

//@ proc-macro:derive_nothing.rs
//@ revisions:rpass1 rpass2
//@ compile-flags: -Zquery-dep-graph -Zcache-proc-macros
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

#[macro_use]
extern crate derive_nothing;

#[cfg(any(rpass1, rpass2))]
#[rustc_clean(cfg = "rpass2", loaded_from_disk = "derive_macro_expansion")]
#[derive(Nothing)]
pub struct Foo;

fn main() {}
