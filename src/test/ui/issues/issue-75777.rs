// Regression test for #75777.
// Checks that a boxed future can be properly constructed.

#![feature(future_readiness_fns)]

use std::future::{self, Future};
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a + Send>>;

fn inject<'a, Env: 'a, A: 'a + Send>(v: A) -> Box<dyn FnOnce(&'a Env) -> BoxFuture<'a, A>> {
    let fut: BoxFuture<'a, A> = Box::pin(future::ready(v));
    Box::new(move |_| fut)
    //~^ ERROR: cannot infer an appropriate lifetime
}

fn main() {}
