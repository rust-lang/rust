//@ run-pass
//@ check-run-results

#![allow(dead_code)]
#![allow(supertrait_item_shadowing_definition)]

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
    println!("{}", i32::CONST);
}
