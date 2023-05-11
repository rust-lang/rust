// Test that a type which is covariant with respect to its region
// parameter yields an error when used in a contravariant way.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

// `S` is contravariant with respect to both parameters.
struct S<'a, 'b> {
    f: &'a isize,
    g: &'b isize,
}

fn use_<'short,'long>(c: S<'long, 'short>,
                      s: &'short isize,
                      l: &'long isize,
                      _where:Option<&'short &'long ()>) {

    let _: S<'long, 'short> = c; // OK
    let _: S<'short, 'short> = c; // OK

    // Test whether S<_,'short> <: S<_,'long>.  Since
    // 'short <= 'long, this would be true if the Contravariant type were
    // covariant with respect to its parameter 'a.

    let _: S<'long, 'long> = c;
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
