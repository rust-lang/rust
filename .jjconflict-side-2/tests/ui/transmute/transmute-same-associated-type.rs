//! Verify transmuting is allowed when `Src` and `Dst` are the same associated type.

//@ check-pass

trait Foo {
    type Bar;
}

unsafe fn noop<F: Foo>(foo: F::Bar) -> F::Bar {
    ::std::mem::transmute(foo)
}

fn main() {}
