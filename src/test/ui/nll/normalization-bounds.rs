// Check that lifetime bounds get checked the right way around with NLL enabled.

//run-pass

trait Visitor<'d> {
    type Value;
}

impl<'a, 'd: 'a> Visitor<'d> for &'a () {
    type Value = ();
}

fn visit_seq<'d: 'a, 'a>() -> <&'a () as Visitor<'d>>::Value {}

fn main() {}
