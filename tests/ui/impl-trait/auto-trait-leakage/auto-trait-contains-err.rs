//@ edition: 2021

use std::future::Future;

fn invalid_future() -> impl Future {}
//~^ ERROR `()` is not a future

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

fn main() {}
