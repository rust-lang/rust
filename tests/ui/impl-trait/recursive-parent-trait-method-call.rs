//! This test checks that we can resolve the `boxed` method call to `FutureExt`,
//! because we know that the anonymous future does not implement `StreamExt`.

//@ edition: 2021
//@ check-pass

use std::future::Future;
use std::pin::Pin;

trait FutureExt: Future + Sized + Send + 'static {
    fn boxed(self) -> Pin<Box<dyn Future<Output = Self::Output> + Send + 'static>> {
        Box::pin(self)
    }
}

trait StreamExt: Future + Sized + Send + 'static {
    fn boxed(self) -> Pin<Box<dyn Future<Output = Self::Output> + Send + 'static>> {
        Box::pin(self)
    }
}

impl<T: Future + Sized + Send + 'static> FutureExt for T {}

fn go(i: usize) -> impl Future<Output = ()> + Send + 'static {
    async move {
        if i != 0 {
            spawn(async move {
                let fut = go(i - 1).boxed();
                fut.await;
            })
            .await;
        }
    }
}

pub fn spawn<T: Send>(
    _: impl Future<Output = T> + Send + 'static,
) -> impl Future<Output = ()> + Send + 'static {
    async move { todo!() }
}

fn main() {}
