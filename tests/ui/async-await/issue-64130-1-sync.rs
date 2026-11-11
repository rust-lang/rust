#![feature(negative_impls)]
//@ edition:2018

// This tests the specialized async-await-specific error when futures don't implement an
// auto trait (which is specifically Sync) due to some type that was captured.

struct Foo;

impl !Sync for Foo {}

fn is_sync<T: Sync>(t: T) { }

async fn bar() {
    let x = Foo;
    baz().await;
    drop(x);
}

async fn baz() { }

fn main() {
    is_sync(bar());
    //~^ ERROR future cannot be shared between threads safely
}
