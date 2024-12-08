#![feature(generic_const_exprs)]
#![feature(specialization)]
#![allow(incomplete_features)]

//--------------------------------------------------

trait Depth {
    const C: usize;
}

trait Type {
    type AT: Depth;
}

//--------------------------------------------------

enum Predicate<const B: bool> {}

trait Satisfied {}

impl Satisfied for Predicate<true> {}

//--------------------------------------------------

trait Spec1 {}

impl<T: Type> Spec1 for T where Predicate<{T::AT::C > 0}>: Satisfied {}

trait Spec2 {}

//impl<T: Type > Spec2 for T where Predicate<{T::AT::C > 1}>: Satisfied {}
impl<T: Type > Spec2 for T where Predicate<true>: Satisfied {}

//--------------------------------------------------

trait Foo {
    fn Bar();
}

impl<T: Spec1> Foo for T {
    default fn Bar() {}
}

impl<T: Spec2> Foo for T {
//~^ ERROR conflicting implementations of trait
    fn Bar() {}
}

fn main() {}
