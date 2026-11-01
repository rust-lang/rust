// Regression test for https://github.com/rust-lang/rust/issues/151297.
//~^ ERROR overflow evaluating the requirement `generator_sig::{opaque#0} == _`
//@ compile-flags: -Znext-solver -Zvalidate-mir
#![feature(coroutines, yield_expr)]

trait Quux {}

fn foo() -> impl Quux { //~ ERROR cannot resolve opaque type
    struct Foo<T>(T);
    impl<T> Quux for Foo<T> {}
    Foo(bar())
}

fn bar() -> impl Quux { //~ ERROR cannot resolve opaque type
    struct Bar<T>(T);
    impl<T> Quux for Bar<T> {}
    Bar(foo())
}

fn ptr() -> impl Sized { //~ ERROR cannot resolve opaque type
    &generator_sig() as *const _
}

fn generator_sig() -> impl Sized { //~ ERROR cannot resolve opaque type
    ptr()
}

fn generator_hold() -> impl Sized {
    #[coroutine]
    || yield generator_sig()
}

fn main() {}
