//! Regression test for <https://github.com/rust-lang/rust/issues/52140>.
//@ run-pass
//@ aux-build:reexport-crate.rs
//@ compile-flags:--extern reexport_crate
//@ edition:2018

mod foo {
    pub use reexport_crate;
}

fn main() {
    ::reexport_crate::hello();
    foo::reexport_crate::hello();
}
