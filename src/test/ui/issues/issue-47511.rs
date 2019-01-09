// Regression test for #47511: anonymous lifetimes can appear
// unconstrained in a return type, but only if they appear just once
// in the input, as the input to a projection.

fn f(_: X) -> X {
    //~^ ERROR return type references an anonymous lifetime
    unimplemented!()
}

fn g<'a>(_: X<'a>) -> X<'a> {
    //~^ ERROR return type references lifetime `'a`, which is not constrained
    unimplemented!()
}

type X<'a> = <&'a () as Trait>::Value;

trait Trait {
    type Value;
}

impl<'a> Trait for &'a () {
    type Value = ();
}

fn main() {}
