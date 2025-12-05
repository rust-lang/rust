// Check that we error when a bound from the impl is not satisfied when
// normalizing an associated type.

trait Visitor<'d> {
    type Value;
}

impl<'a, 'd: 'a> Visitor<'d> for &'a () {
    type Value = ();
}

fn visit_seq<'d, 'a: 'd>() -> <&'a () as Visitor<'d>>::Value {}
//~^ ERROR lifetime may not live long enough
//~| ERROR cannot infer

fn main() {}
