//! Regression test for <https://github.com/rust-lang/rust/issues/52705>.
//! Type didn't resolve when module name was path's root.
//@ run-pass
#![allow(dead_code)]
//@ aux-build:direct-renamed-extern-crate.rs
//@ compile-flags:--extern direct_renamed_extern_crate
//@ edition:2018

mod png {
    use direct_renamed_extern_crate as png_ext;

    fn foo() -> png_ext::DecodingError { unimplemented!() }
}

fn main() {
    println!("Hello, world!");
}
