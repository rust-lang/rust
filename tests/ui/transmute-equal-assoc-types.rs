//@ check-pass

trait Foo {
    type Bar;
}

unsafe fn noop<F: Foo>(foo: F::Bar) -> F::Bar {
    ::std::mem::transmute(foo)
}

fn main() {}
