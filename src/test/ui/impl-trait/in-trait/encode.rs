// build-pass
// compile-flags: --crate-type=lib

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Foo {
    fn bar() -> impl Sized;
}
