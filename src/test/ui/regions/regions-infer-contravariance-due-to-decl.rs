// Test that a type which is contravariant with respect to its region
// parameter yields an error when used in a covariant way.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

use std::marker;

// This is contravariant with respect to 'a, meaning that
// Contravariant<'foo> <: Contravariant<'static> because
// 'foo <= 'static
struct Contravariant<'a> {
    marker: marker::PhantomData<&'a()>
}

fn use_<'short,'long>(c: Contravariant<'short>,
                      s: &'short isize,
                      l: &'long isize,
                      _where:Option<&'short &'long ()>) {

    // Test whether Contravariant<'short> <: Contravariant<'long>.  Since
    // 'short <= 'long, this would be true if the Contravariant type were
    // covariant with respect to its parameter 'a.

    let _: Contravariant<'long> = c; //~ ERROR E0623
}

fn main() {}
