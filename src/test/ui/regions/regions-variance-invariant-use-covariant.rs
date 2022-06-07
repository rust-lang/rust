// Test that a type which is invariant with respect to its region
// parameter used in a covariant way yields an error.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

struct Invariant<'a> {
    f: &'a mut &'a isize
}

fn use_<'b>(c: Invariant<'b>) {

    // For this assignment to be legal, Invariant<'b> <: Invariant<'static>.
    // Since 'b <= 'static, this would be true if Invariant were covariant
    // with respect to its parameter 'a.

    let _: Invariant<'static> = c;
    //~^ ERROR lifetime may not live long enough
}

fn main() { }
