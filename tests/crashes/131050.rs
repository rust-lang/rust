//@ known-bug: #131050
//@ compile-flags: --edition=2021

use std::future::Future;

fn invalid_future() -> impl Future {}

fn create_complex_future() -> impl Future<Output = impl ReturnsSend> {
    async { &|| async { invalid_future().await } }
}

fn coerce_impl_trait() -> impl Future<Output = impl Send> {
    create_complex_future()
}

trait ReturnsSend {}

impl<F, R> ReturnsSend for F
where
    F: Fn() -> R,
    R: Send,
{
}
