// edition:2018
#![allow(dead_code)]
use std::future::Future;
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
//   ^^^^^^^^^ This would come from the `futures` crate in real code.

fn foo<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
    // We could instead use an `async` block, but this way we have no std spans.
    x //~ ERROR mismatched types
}
fn bar<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
    Box::new(x) //~ ERROR mismatched types
    //~^ HELP use `Box::pin`
}
fn baz<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
    Pin::new(x) //~ ERROR mismatched types
    //~^ ERROR the trait bound
}
fn qux<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
    Pin::new(Box::new(x)) //~ ERROR mismatched types
    //~^ ERROR the trait bound
}

fn main() {}
