#![feature(rustc_attrs)]

#[rustc_must_implement_one_of(a, b)]
//~^ Function not found in this trait
//~| Function not found in this trait
trait Tr0 {}

#[rustc_must_implement_one_of(a, b)]
//~^ Function not found in this trait
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
    const A: u8 = 1; //~ Not a function

    type B; //~ Not a function
}

#[rustc_must_implement_one_of(a, b)]
trait Tr5 {
    fn a(); //~ This function doesn't have a default implementation

    fn b(); //~ This function doesn't have a default implementation
}

#[rustc_must_implement_one_of(abc, xyz)]
//~^ attribute can only be applied to a trait
fn function() {}

#[rustc_must_implement_one_of(abc, xyz)]
//~^ attribute can only be applied to a trait
struct Struct {}

fn main() {}
