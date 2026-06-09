//@ check-pass
// This resulted in an ICE. Test for future-proofing
// Issue #24227

#![allow(unused)]

struct Foo<'a> {
    x: &'a u8
}

impl<'a> Foo<'a> {
    fn foo() {
        let mut tmp: Self;
    }

}

fn main() {}
