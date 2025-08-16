// This test tests that derive-macro execution is cached.
// HOWEVER, this test can currently only be checked manually,
// by running it (through compiletest) with `-- --nocapture --verbose`.
// The proc-macro (for `Nothing`) prints a message to stderr when invoked,
// and this message should only be present during the first invocation,
// because the cached result should be used for the second invocation.
// FIXME(pr-time): Properly have the test check this, but how? UI-test that tests for `.stderr`?

//@ aux-build:derive_nothing.rs
//@ revisions:rpass1 rpass2
//@ compile-flags: -Zquery-dep-graph -Zcache-derive-macros

#![feature(rustc_attrs)]

#[macro_use]
extern crate derive_nothing;

#[cfg(rpass1)]
#[derive(Nothing)]
pub struct Foo;

#[cfg(rpass2)]
#[derive(Nothing)]
#[rustc_clean(cfg = "rpass2", loaded_from_disk = "derive_macro_expansion")]
pub struct Foo;

fn main() {}
