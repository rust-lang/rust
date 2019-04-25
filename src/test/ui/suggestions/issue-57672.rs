// aux-build:foo.rs
// compile-flags:--extern foo
// edition:2018

#![deny(unused_extern_crates)]

extern crate foo as foo_renamed;
//~^ ERROR `extern crate` is not idiomatic in the new edition

pub mod m {
    pub use foo_renamed::Foo;
}

fn main() {}
