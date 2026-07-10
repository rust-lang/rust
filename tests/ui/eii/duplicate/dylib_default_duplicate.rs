//@ aux-build: dylib_default.rs
//@ needs-crate-type: dylib
//@ compile-flags: --emit link
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Regression test for https://github.com/rust-lang/rust/issues/156320.
// A default implementation from an upstream dylib has already been selected and
// must not be overridden by a downstream explicit implementation.
#![feature(extern_item_impls)]

extern crate dylib_default;

#[unsafe(dylib_default::eii1)]
fn other(x: u64) {
    //~^ ERROR multiple implementations of `#[eii1]`
    println!("1{x}");
}

fn main() {}
