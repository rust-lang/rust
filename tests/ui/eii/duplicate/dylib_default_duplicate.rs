//@ aux-build: dylib_default.rs
//@ needs-crate-type: dylib
//@ compile-flags: --emit link
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Regression test for https://github.com/rust-lang/rust/issues/156320.
// A default implementation from an upstream dylib has already been selected and
// must not be overridden by a downstream explicit implementation.
#![feature(extern_item_impls)]

extern crate dylib_default;

#[dylib_default::eii1]
fn other(x: u64) {
    //~^ ERROR multiple implementations of `#[eii1]`
    println!("1{x}");
}

fn main() {}
