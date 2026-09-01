//@ edition: 2021
//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/146813>.
// Used to detect cycle when type-checking `fails`

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
