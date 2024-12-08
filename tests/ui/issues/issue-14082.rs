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
