//@ run-pass
//@ aux-build:pub_use_xcrate1.rs
//@ aux-build:pub_use_xcrate2.rs


extern crate pub_use_xcrate2;

use pub_use_xcrate2::Foo;

pub fn main() {
    let _foo: Foo = Foo {
        name: 0
    };
}
