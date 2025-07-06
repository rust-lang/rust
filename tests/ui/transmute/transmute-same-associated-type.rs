//! This test verifies that `std::mem::transmute` is allowed between two values
//! of the exact same associated type (`F::Bar`).

//@ check-pass

trait Foo {
    type Bar;
}

unsafe fn noop<F: Foo>(foo: F::Bar) -> F::Bar {
    ::std::mem::transmute(foo)
}

fn main() {}
