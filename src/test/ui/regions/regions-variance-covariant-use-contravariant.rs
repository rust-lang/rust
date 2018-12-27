// Test that a type which is covariant with respect to its region
// parameter yields an error when used in a contravariant way.
//
// Note: see variance-regions-*.rs for the tests that check that the
// variance inference works in the first place.

// This is covariant with respect to 'a, meaning that
// Covariant<'foo> <: Covariant<'static> because
// 'foo <= 'static
struct Covariant<'a> {
    f: extern "Rust" fn(&'a isize)
}

fn use_<'short,'long>(c: Covariant<'long>,
                      s: &'short isize,
                      l: &'long isize,
                      _where:Option<&'short &'long ()>) {

    // Test whether Covariant<'long> <: Covariant<'short>.  Since
    // 'short <= 'long, this would be true if the Covariant type were
    // contravariant with respect to its parameter 'a.

    let _: Covariant<'short> = c; //~ ERROR E0623
}

fn main() {}
