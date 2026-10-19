//@ revisions: edition2018 edition2021
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021..

//! Ideally, this would compile.

use std::cell::Cell;
use std::marker::PhantomData;
use std::panic::UnwindSafe;

#[derive(Clone, Copy)]
struct Foo(PhantomData<Cell<()>>);

fn main() {
    let foo = Foo(PhantomData);
    assert_auto_traits(|| {
        //~^ ERROR `Cell<()>` cannot be shared between threads safely
        //~| ERROR the type `UnsafeCell<()>` may contain interior mutability and a reference may not be safely transferable across a catch_unwind boundary
        let _foo = foo;
    });
}

fn assert_auto_traits(_: impl Fn() + Send + UnwindSafe) {}
