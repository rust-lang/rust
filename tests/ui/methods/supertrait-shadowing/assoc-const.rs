//@ run-pass

#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

trait A {
    const CONST: i32;
}
impl<T> A for T {
    const CONST: i32 = 1;
}

trait B: A {
    const CONST: i32;
}
impl<T> B for T {
    const CONST: i32 = 2;
}

fn main() {
    assert_eq!(i32::CONST, 2)
}
