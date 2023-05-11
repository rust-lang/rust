// build-pass
// compile-flags: --crate-type=lib
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Foo {
    fn bar() -> impl Sized;
}
