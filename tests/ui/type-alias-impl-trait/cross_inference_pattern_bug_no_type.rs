//@ compile-flags: --crate-type=lib
//@ edition: 2021
//@ rustc-env:RUST_BACKTRACE=0
//@ check-pass

// tracked in https://github.com/rust-lang/rust/issues/96572

#![feature(type_alias_impl_trait)]

fn main() {
    type T = impl Copy;
    let foo: T = (1u32, 2u32);
    let (a, b) = foo; // this line used to make the code fail
}
