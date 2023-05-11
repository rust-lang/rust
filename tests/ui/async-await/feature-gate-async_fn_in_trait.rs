// edition:2021

// RPITIT is not enough to allow use of async functions
#![allow(incomplete_features)]
#![feature(return_position_impl_trait_in_trait)]

trait T {
    async fn foo(); //~ ERROR functions in traits cannot be declared `async`
}

// Both return_position_impl_trait_in_trait and async_fn_in_trait are required for this (see also
// feature-gate-return_position_impl_trait_in_trait.rs)
trait T2 {
    async fn foo() -> impl Sized; //~ ERROR functions in traits cannot be declared `async`
}

trait T3 {
    fn foo() -> impl std::future::Future<Output = ()>;
}

impl T3 for () {
    async fn foo() {} //~ ERROR functions in traits cannot be declared `async`
}

fn main() {}
