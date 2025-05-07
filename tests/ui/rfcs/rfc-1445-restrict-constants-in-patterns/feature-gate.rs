// Test that use of structural-match traits is only permitted with a feature gate,
// and that if a feature gate is supplied, it permits the type to be
// used in a match.

//@ revisions: with_gate no_gate
//@[with_gate] check-pass
// gate-test-structural_match

#![allow(unused)]
#![cfg_attr(with_gate, feature(structural_match))]


struct Foo {
    x: u32
}

const FOO: Foo = Foo { x: 0 };

fn main() {
    let y = Foo { x: 1 };
    match y {
        FOO => { }
        _ => { }
    }
}

impl std::marker::StructuralPartialEq for Foo { }
//[no_gate]~^ ERROR use of unstable library feature `structural_match`

impl PartialEq<Foo> for Foo {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
    }
}
impl Eq for Foo { }
