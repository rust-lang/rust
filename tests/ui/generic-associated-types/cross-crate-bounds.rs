// regression test for #73816
// We handled bounds differently when `feature(generic_associated_types)` was enabled

//@ edition:2018
//@ aux-build:foo_defn.rs

extern crate foo_defn;

use foo_defn::Foo;
use std::{future::Future, pin::Pin};

pub struct FooImpl;

impl Foo for FooImpl {
    type Bar = ();
    //~^ ERROR the trait bound `(): AsRef<()>` is not satisfied
    fn foo(&self) -> Pin<Box<dyn Future<Output = Self::Bar> + '_>> {
        panic!()
    }
}

async fn foo() {
    bar(&FooImpl).await;
}

async fn bar<F: Foo>(foo: &F) {
    foo.foo().await.as_ref();
}

fn main() {
    // futures::executor::block_on(foo());
}
