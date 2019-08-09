// Provoke an unresolved type error (T).
// Error message should pinpoint the type parameter T as needing to be bound
// (rather than give a general error message)
// edition:2018
#![feature(async_await)]
async fn bar<T>() -> () {}

async fn foo() {
    bar().await;
    //~^ ERROR type inside `async` object must be known in this context
    //~| NOTE cannot infer type for `T`
    //~| NOTE the type is part of the `async` object because of this `await`
    //~| NOTE in this expansion of desugaring of `await`
}
fn main() {}
