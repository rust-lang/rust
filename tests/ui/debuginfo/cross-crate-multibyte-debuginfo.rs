//! Regression test for <https://github.com/rust-lang/rust/issues/24687>.
//@ run-pass
//@ aux-build:cross-crate-multibyte-debuginfo.rs
//@ compile-flags:-g

extern crate cross_crate_multibyte_debuginfo as d;

fn main() {
    // Create a `D`, which has a destructor whose body will be codegen'ed
    // into the generated code here, and thus the local debuginfo will
    // need references into the original source locations from
    // `importer` above.
    let _d = d::D("Hi");
}
