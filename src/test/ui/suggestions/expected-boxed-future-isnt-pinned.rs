// edition:2018
// run-rustfix
#![allow(dead_code)]
use std::future::Future;
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
//   ^^^^^^^^^ This would come from the `futures` crate in real code.

fn foo<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
    // We could instead use an `async` block, but this way we have no std spans.
    x //~ ERROR mismatched types
}

fn main() {}
