//@ aux-build:codegen3.rs
// Tests whether name resulution respects privacy properly.
#![feature(extern_item_impls)]

extern crate codegen3 as codegen;

// has a span but in the other crate
//~? ERROR `#[eii2]` required, but not found
//~? ERROR `#[eii3]` required, but not found

#[codegen::eii1]
fn eii1_impl(x: u64) {
    println!("{x:?}")
}

#[codegen::eii3] //~ ERROR failed to resolve: could not find `eii3` in `codegen`
fn eii3_impl(x: u64) {
    println!("{x:?}")
}

// what you would write:
fn main() {
    // directly
    eii1_impl(21);
    // through the alias
    codegen::decl1(42); //~ ERROR function `decl1` is private
}
