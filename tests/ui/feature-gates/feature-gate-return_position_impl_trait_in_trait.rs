// edition:2021

// async_fn_in_trait is not enough to allow use of RPITIT
#![allow(incomplete_features)]
#![feature(async_fn_in_trait)]

trait Foo {
    fn bar() -> impl Sized; //~ ERROR `impl Trait` only allowed in function and inherent method return types, not in trait method return
    fn baz() -> Box<impl std::fmt::Display>; //~ ERROR `impl Trait` only allowed in function and inherent method return types, not in trait method return
}

// Both return_position_impl_trait_in_trait and async_fn_in_trait are required for this (see also
// feature-gate-async_fn_in_trait.rs)
trait AsyncFoo {
    async fn bar() -> impl Sized; //~ ERROR `impl Trait` only allowed in function and inherent method return types, not in trait method return
}

fn main() {}
