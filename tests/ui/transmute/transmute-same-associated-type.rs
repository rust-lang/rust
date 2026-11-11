//! Verify transmuting is allowed when `Src` and `Dst` are the same associated type.
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ check-pass

trait Foo {
    type Bar;
}

unsafe fn noop<F: Foo>(foo: F::Bar) -> F::Bar {
    ::std::mem::transmute(foo)
}

fn main() {}
