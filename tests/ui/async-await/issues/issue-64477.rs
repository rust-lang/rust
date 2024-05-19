// Regression test for #64477.
//
// We were incorrectly claiming that the `f(x).await` future captured
// a value of type `T`, and hence that `T: Send` would have to hold.
//
//@ check-pass
//@ edition:2018

use std::future::Future;
use std::pin::Pin;

fn f<T>(_: &T) -> Pin<Box<dyn Future<Output = ()> + Send>> {
    unimplemented!()
}

pub fn g<T: Sync>(x: &'static T) -> impl Future<Output = ()> + Send {
    async move { f(x).await }
}

fn main() { }
