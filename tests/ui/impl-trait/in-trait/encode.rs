//@ build-pass
//@ compile-flags: --crate-type=lib

#![allow(incomplete_features)]

trait Foo {
    fn bar() -> impl Sized;
}
