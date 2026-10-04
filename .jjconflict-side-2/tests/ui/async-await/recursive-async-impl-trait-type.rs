//@ edition:2018
// Test that impl trait does not allow creating recursive types that are
// otherwise forbidden when using `async` and `await`.

async fn recursive_async_function() -> () {
    //~^ ERROR recursion in an async fn requires boxing
    recursive_async_function().await;
}

fn main() {}
