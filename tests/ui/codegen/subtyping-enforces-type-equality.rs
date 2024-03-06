//@ ignore-pass
//@ build-pass
//@ edition:2021
use std::future::Future;
use std::pin::Pin;

type BoxFuture<T> = Pin<Box<dyn Future<Output = T>>>;

fn main() {
    let _ = wrapper_call(handler);
}

async fn wrapper_call(handler: impl Handler) {
    handler.call().await;
}
async fn handler() {
    f(&()).await;
}
async fn f<'a>(db: impl Acquire<'a>) {
    db.acquire().await;
}

trait Handler {
    type Future: Future;
    fn call(self) -> Self::Future;
}

impl<Fut, F> Handler for F
where
    F: Fn() -> Fut,
    Fut: Future,
{
    type Future = Fut;
    fn call(self) -> Self::Future {
        loop {}
    }
}

trait Acquire<'a> {
    type Connection;
    fn acquire(self) -> BoxFuture<Self::Connection>;
}
impl<'a> Acquire<'a> for &'a () {
    type Connection = Self;
    fn acquire(self) -> BoxFuture<Self> {
        loop {}
    }
}
