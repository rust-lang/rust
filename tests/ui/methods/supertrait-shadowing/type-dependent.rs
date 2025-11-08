//@ run-pass
//@ check-run-results

// Makes sure we can shadow with type-dependent method syntax.

#![allow(dead_code)]
#![allow(supertrait_item_shadowing_definition)]

trait A {
    fn hello() {
        println!("A");
    }
}
impl<T> A for T {}

trait B: A {
    fn hello() {
        println!("B");
    }
}
impl<T> B for T {}

fn foo<T>() {
    T::hello();
}

fn main() {
    foo::<()>();
}
