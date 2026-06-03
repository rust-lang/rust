//@ run-pass

// Makes sure we can shadow with type-dependent method syntax.

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

trait A {
    fn hello() -> &'static str {
        "A"
    }
}
impl<T> A for T {}

trait B: A {
    fn hello() -> &'static str {
        "B"
    }
}
impl<T> B for T {}

fn foo<T>() -> &'static str {
    T::hello()
}

fn main() {
    assert_eq!(foo::<()>(), "B");
}
