#![feature(const_generics)]
#![allow(incomplete_features)]

trait Foo {
    const VAL: usize;
}

trait MyTrait {}

trait True {}
struct Is<const T: bool>;
impl True for Is<{true}> {}

impl<T: Foo> MyTrait for T where Is<{T::VAL == 5}>: True {}
//~^ ERROR constant expression depends on a generic parameter
impl<T: Foo> MyTrait for T where Is<{T::VAL == 6}>: True {}
//~^ ERROR constant expression depends on a generic parameter

fn main() {}
