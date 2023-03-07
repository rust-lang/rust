// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// Provoke an unresolved type error (T).
// Error message should pinpoint the type parameter T as needing to be bound
// (rather than give a general error message)
// edition:2018

async fn bar<T>() -> () {}

async fn foo() {
    bar().await;
    //[drop_tracking_mir]~^ ERROR type annotations needed
    //[drop_tracking_mir]~| NOTE cannot infer type of the type parameter `T`
    //[no_drop_tracking,drop_tracking]~^^^ ERROR type inside `async fn` body must be known in this context
    //[no_drop_tracking,drop_tracking]~| ERROR type inside `async fn` body must be known in this context
    //[no_drop_tracking,drop_tracking]~| ERROR type inside `async fn` body must be known in this context
    //[no_drop_tracking,drop_tracking]~| NOTE cannot infer type for type parameter `T`
    //[no_drop_tracking,drop_tracking]~| NOTE cannot infer type for type parameter `T`
    //[no_drop_tracking,drop_tracking]~| NOTE cannot infer type for type parameter `T`
    //[no_drop_tracking,drop_tracking]~| NOTE the type is part of the `async fn` body because of this `await`
    //[no_drop_tracking,drop_tracking]~| NOTE the type is part of the `async fn` body because of this `await`
    //[no_drop_tracking,drop_tracking]~| NOTE the type is part of the `async fn` body because of this `await`
    //[no_drop_tracking,drop_tracking]~| NOTE in this expansion of desugaring of `await`
    //[no_drop_tracking,drop_tracking]~| NOTE in this expansion of desugaring of `await`
    //[no_drop_tracking,drop_tracking]~| NOTE in this expansion of desugaring of `await`
    //[no_drop_tracking]~^^^^^^^^^^^^^^^ ERROR type inside `async fn` body must be known in this context
    //[no_drop_tracking]~| ERROR type inside `async fn` body must be known in this context
    //[no_drop_tracking]~| NOTE cannot infer type for type parameter `T`
    //[no_drop_tracking]~| NOTE cannot infer type for type parameter `T`
    //[no_drop_tracking]~| NOTE the type is part of the `async fn` body because of this `await`
    //[no_drop_tracking]~| NOTE the type is part of the `async fn` body because of this `await`
    //[no_drop_tracking]~| NOTE in this expansion of desugaring of `await`
    //[no_drop_tracking]~| NOTE in this expansion of desugaring of `await`
}
fn main() {}
