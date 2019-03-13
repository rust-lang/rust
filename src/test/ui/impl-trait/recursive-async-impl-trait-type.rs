// edition:2018
// Test that impl trait does not allow creating recursive types that are
// otherwise forbidden when using `async` and `await`.

#![feature(await_macro, async_await, futures_api, generators)]

async fn recursive_async_function() -> () { //~ ERROR
    await!(recursive_async_function());
}

fn main() {}
