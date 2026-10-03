//@ edition:2018..2021

use std::marker::{PhantomData, PhantomPinned};

#[derive(Clone, Copy)]
struct Foo(PhantomData<*const ()>, PhantomPinned);

unsafe impl Sync for Foo {}

fn main() {
    let foo = Foo(PhantomData, PhantomPinned);
    assert_auto_traits(|| {
        //~^ ERROR `PhantomPinned` cannot be unpinned
        //~| ERROR `*const ()` cannot be sent between threads safely
        let _foo = foo;
    });
}

fn assert_auto_traits(_: impl Fn() + Send + Unpin) {}
