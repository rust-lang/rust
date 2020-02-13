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

// FIXME: uncomment these once this commit is in Beta and we can rely on `rustc_on_unimplemented`
//        having filtering for `Self` being a trait.
//
// fn bar<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
//     Box::new(x)
// }
//
// fn baz<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
//     Pin::new(x)
// }
//
// fn qux<F: Future<Output=i32> + Send + 'static>(x: F) -> BoxFuture<'static, i32> {
//     Pin::new(Box::new(x))
// }

fn main() {}
