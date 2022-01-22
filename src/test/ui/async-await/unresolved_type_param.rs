// Provoke an unresolved type error (T).
// Error message should pinpoint the type parameter T as needing to be bound
// (rather than give a general error message)
// edition:2018

// FIXME(eholk): temporarily disabled while drop range tracking is disabled
// (see generator_interior.rs:27)
// ignore-test

async fn bar<T>() -> () {}

async fn foo() {
    bar().await;
    //~^ ERROR type inside `async fn` body must be known in this context
    //~| ERROR type inside `async fn` body must be known in this context
    //~| ERROR type inside `async fn` body must be known in this context
    //~| NOTE cannot infer type for type parameter `T`
    //~| NOTE cannot infer type for type parameter `T`
    //~| NOTE cannot infer type for type parameter `T`
    //~| NOTE the type is part of the `async fn` body because of this `await`
    //~| NOTE the type is part of the `async fn` body because of this `await`
    //~| NOTE the type is part of the `async fn` body because of this `await`
    //~| NOTE in this expansion of desugaring of `await`
    //~| NOTE in this expansion of desugaring of `await`
    //~| NOTE in this expansion of desugaring of `await`
}
fn main() {}
