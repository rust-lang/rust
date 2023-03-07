#![feature(rustc_attrs)]

#[rustc_must_implement_one_of(a, b)]
//~^ function not found in this trait
//~| function not found in this trait
trait Tr0 {}

#[rustc_must_implement_one_of(a, b)]
//~^ function not found in this trait
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
    const A: u8 = 1; //~ not a function

    type B; //~ not a function
}

#[rustc_must_implement_one_of(a, b)]
trait Tr5 {
    fn a(); //~ function doesn't have a default implementation

    fn b(); //~ function doesn't have a default implementation
}

#[rustc_must_implement_one_of(abc, xyz)]
//~^ attribute should be applied to a trait
fn function() {}

#[rustc_must_implement_one_of(abc, xyz)]
//~^ attribute should be applied to a trait
struct Struct {}

fn main() {}
