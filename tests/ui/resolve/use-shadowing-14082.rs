//! Regression test for https://github.com/rust-lang/rust/issues/14082

//@ check-pass

#![allow(unused_imports, dead_code)]

use foo::Foo;

mod foo {
    pub use m::Foo; // this should shadow d::Foo
}

mod m {
    pub struct Foo;
}

mod d {
    pub struct Foo;
}

fn main() {}
