#![feature(rustc_attrs)]

#[rustc_must_implement_one_of(a, b)]
//~^ Method not found in this trait
//~| Method not found in this trait
trait Tr0 {}

#[rustc_must_implement_one_of(a, b)]
//~^ Method not found in this trait
trait Tr1 {
    fn a() {}
}

#[rustc_must_implement_one_of(a)]
//~^ the `#[rustc_must_implement_one_of]` attribute must be used with at least 2 args
trait Tr2 {
    fn a() {}
}

#[rustc_must_implement_one_of]
//~^ malformed `rustc_must_implement_one_of` attribute input
trait Tr3 {}

#[rustc_must_implement_one_of(A, B)]
trait Tr4 {
    const A: u8 = 1; //~ Not a method

    type B; //~ Not a method
}

#[rustc_must_implement_one_of(a, b)]
trait Tr5 {
    fn a(); //~ This method doesn't have a default implementation

    fn b(); //~ This method doesn't have a default implementation
}

fn main() {}
