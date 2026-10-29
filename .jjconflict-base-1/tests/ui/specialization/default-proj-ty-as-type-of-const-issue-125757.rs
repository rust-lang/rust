#![feature(specialization)]
#![allow(incomplete_features)]

trait Trait {
    type Type;
}

impl Trait for i32 {
    default type Type = i32;
}

struct Wrapper<const C: <i32 as Trait>::Type> {}
//~^ ERROR `<i32 as Trait>::Type` is forbidden as the type of a const generic parameter

impl<const C: usize> Wrapper<C> {}
//~^ ERROR the constant `C` is not of type `<i32 as Trait>::Type`

fn main() {}
