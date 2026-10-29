//@ aux-build:foo.rs
//@ compile-flags:--extern foo
//@ check-pass
//@ edition:2018

#![deny(unused_extern_crates)]

extern crate foo as foo_renamed;

pub mod m {
    pub use foo_renamed::Foo;
}

fn main() {}
