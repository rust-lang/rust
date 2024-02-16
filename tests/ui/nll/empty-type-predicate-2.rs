// Regression test for #65553
//
// `D::Error:` is lowered to `D::Error: ReEmpty` - check that we don't ICE in
// NLL for the unexpected region.

//@ check-pass

trait Deserializer {
    type Error;
}

fn d1<D: Deserializer>() where D::Error: {}

fn d2<D: Deserializer>() {
    d1::<D>();
}

fn main() {}
