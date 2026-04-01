//@ known-bug: #140577
//@ compile-flags: -Znext-solver=globally
//@ edition:2021

use std::future::Future;
use std::pin::Pin;
trait Acquire {
    type Connection;
}
impl Acquire for &'static () {
    type Connection = ();
}
fn b<T: Acquire>() -> impl Future + Send {
    let x: Pin<Box<dyn Future<Output = T::Connection> + Send>> = todo!();
    x
}
fn main() {
    async {
        b::<&()>().await;
    }
    .aa();
}

impl<F> Filter for F where F: Send {}

trait Filter {
    fn aa(self)
    where
        Self: Sized,
    {
    }
}
