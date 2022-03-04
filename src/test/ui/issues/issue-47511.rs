// check-fail
// known-bug

// Regression test for #47511: anonymous lifetimes can appear
// unconstrained in a return type, but only if they appear just once
// in the input, as the input to a projection.

fn f(_: X) -> X {
    unimplemented!()
}

fn g<'a>(_: X<'a>) -> X<'a> {
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
