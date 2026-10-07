//! Regression test for https://github.com/rust-lang/rust/issues/141850.
//@ edition: 2024
//@ compile-flags: -Znext-solver=globally

#![feature(pin_ergonomics)]
#![allow(incomplete_features, dead_code, unused_must_use)]

async fn a() {
    wrapper_call(handler).await;
}

async fn wrapper_call<F>(_: F) -> F::Output
where
    F: Handler,
{
    todo!()
}

async fn handler(); //~ ERROR free function without a body

trait Handler {
    type Output;
}

impl<Func, Fut> Handler for Func
where
    Func: Fn() -> Fut,
    Fut: Future,
{
    type Output = Fut;
}

fn main() {}
