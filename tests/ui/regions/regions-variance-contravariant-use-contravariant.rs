//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that a type which is contravariant with respect to its region
// parameter compiles successfully when used in a contravariant way.
//
// Note: see ui/variance/variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.


struct Contravariant<'a> {
    f: &'a isize
}

fn use_<'a>(c: Contravariant<'a>) {
    let x = 3;

    // 'b winds up being inferred to this call.
    // Contravariant<'a> <: Contravariant<'call> is true
    // if 'call <= 'a, which is true, so no error.
    collapse(&x, c);

    fn collapse<'b>(x: &'b isize, c: Contravariant<'b>) { }
}

pub fn main() {}
