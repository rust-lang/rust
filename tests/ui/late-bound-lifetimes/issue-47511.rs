//@ check-pass

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
