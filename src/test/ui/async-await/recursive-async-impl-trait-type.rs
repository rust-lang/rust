// edition:2018
// Test that impl trait does not allow creating recursive types that are
// otherwise forbidden when using `async` and `await`.

#![feature(async_await)]

async fn recursive_async_function() -> () { //~ ERROR
    recursive_async_function().await;
}

fn main() {}
