// Test that a type which is covariant with respect to its region
// parameter yields an error when used in a contravariant way.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

// This is contravariant with respect to 'a, meaning that
// Contravariant<'long> <: Contravariant<'short> iff
// 'short <= 'long
struct Contravariant<'a> {
    f: &'a isize
}

fn use_<'short,'long>(c: Contravariant<'short>,
                      s: &'short isize,
                      l: &'long isize,
                      _where:Option<&'short &'long ()>) {

    // Test whether Contravariant<'short> <: Contravariant<'long>.  Since
    // 'short <= 'long, this would be true if the Contravariant type were
    // covariant with respect to its parameter 'a.

    let _: Contravariant<'long> = c;
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
