#![feature(supertrait_item_shadowing)]
#![feature(min_generic_const_args)]

use std::mem::size_of;

trait A {
    fn hello(&self) -> &'static str {
        "A"
    }
    type Assoc;
    const CONST: i32;
}
impl<T> A for T {
    type Assoc = i8;
    const CONST: i32 = 1;
}

trait B {
    fn hello(&self) -> &'static str {
        "B"
    }
    type Assoc;
    const CONST: i32;
}
impl<T> B for T {
    type Assoc = i16;
    const CONST: i32 = 2;
}

fn main() {
    ().hello();
    //~^ ERROR multiple applicable items in scope
    check::<()>();
}

fn check<T: A + B>() {
    let _ = size_of::<T::Assoc>();
    //~^ ERROR ambiguous associated type `Assoc` in bounds of `T`
    let _ = T::CONST;
    //~^ ERROR multiple applicable items in scope
}
