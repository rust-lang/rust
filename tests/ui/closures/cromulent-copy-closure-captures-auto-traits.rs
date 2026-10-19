//@ revisions: edition2018 edition2021
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021..
//@ run-pass

use std::marker::{PhantomData, PhantomPinned};
use std::panic::UnwindSafe;

#[derive(Clone, Copy)]
struct Foo(PhantomData<(*const (), &'static mut ())>, PhantomPinned);

unsafe impl Sync for Foo {}

fn main() {
    let foo = Foo(PhantomData, PhantomPinned);
    assert_auto_traits(|| {
        let _foo = foo;
    });
}

fn assert_auto_traits(_: impl Fn() + Send + Unpin + UnwindSafe) {}
