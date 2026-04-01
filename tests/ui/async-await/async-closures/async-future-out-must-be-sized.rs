//@ edition: 2021

// Ensure that the output of a `fn` pointer that implements `AsyncFn*` is `Sized`,
// like other built-in impls of an fn pointer, like `Fn*`.

use std::future::Future;

fn foo() -> fn() -> dyn Future<Output = ()> {
    todo!()
}

async fn is_async_fn(f: impl AsyncFn()) {
    f().await;
}

fn main() {
    is_async_fn(foo());
    //~^ ERROR the size for values of type `dyn Future<Output = ()>` cannot be known at compilation time
    //~| ERROR the size for values of type `dyn Future<Output = ()>` cannot be known at compilation time
}
