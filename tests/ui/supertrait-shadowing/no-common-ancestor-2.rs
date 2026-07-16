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

trait C: A + B {
    fn hello(&self) -> &'static str {
        "C"
    }
    type Assoc;
    type const CONST: i32;
}
impl<T> C for T {
    type Assoc = i32;
    type const CONST: i32 = 3;
}

// Since `D` is not a subtrait of `C`,
// we have no obvious lower bound.

trait D: B {
    fn hello(&self) -> &'static str {
        "D"
    }
    type Assoc;
    type const CONST: i32;
}
impl<T> D for T {
    type Assoc = i64;
    type const CONST: i32 = 4;
}

fn main() {
    ().hello();
    //~^ ERROR multiple applicable items in scope
    check::<()>();
}

fn check<T: C + D>() {
    let _ = size_of::<T::Assoc>();
    //~^ ERROR ambiguous associated type `Assoc` in bounds of `T`
    let _ = T::CONST;
    //~^ ERROR multiple applicable items in scope
}
