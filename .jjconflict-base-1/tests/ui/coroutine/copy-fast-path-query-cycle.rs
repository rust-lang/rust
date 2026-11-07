//@ edition: 2024
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for #146813. We previously used a pseudo-canonical
// query during HIR typeck which caused a query cycle when looking at the
// witness of a coroutine.

use std::future::Future;

trait ConnectMiddleware {}

trait ConnectHandler: Sized {
    fn with<M>(self, _: M) -> impl ConnectHandler
    where
        M: ConnectMiddleware,
    {
        LayeredConnectHandler
    }
}

struct LayeredConnectHandler;
impl ConnectHandler for LayeredConnectHandler {}
impl<F> ConnectHandler for F where F: FnOnce() {}

impl<F, Fut> ConnectMiddleware for F
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()> + Send,
{
}

pub async fn fails() {
    { || {} }
        .with(async || ())
        .with(async || ())
        .with(async || ());
}
fn main() {}
