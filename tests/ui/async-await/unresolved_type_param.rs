// Provoke an unresolved type error (T).
// Error message should pinpoint the type parameter T as needing to be bound
// (rather than give a general error message)
//@ edition:2018

async fn bar<T>() -> () {}

async fn foo() {
    bar().await;
    //~^ ERROR type annotations needed
}
fn main() {}
