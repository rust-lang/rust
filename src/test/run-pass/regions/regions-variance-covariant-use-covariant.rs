// run-pass
#![allow(dead_code)]
// Test that a type which is covariant with respect to its region
// parameter is successful when used in a covariant way.
//
// Note: see compile-fail/variance-regions-*.rs for the tests that
// check that the variance inference works in the first place.

// This is covariant with respect to 'a, meaning that
// Covariant<'foo> <: Covariant<'static> because
// 'foo <= 'static
// pretty-expanded FIXME #23616

struct Covariant<'a> {
    f: extern "Rust" fn(&'a isize)
}

fn use_<'a>(c: Covariant<'a>) {
    // OK Because Covariant<'a> <: Covariant<'static> iff 'a <= 'static
    let _: Covariant<'static> = c;
}

pub fn main() {}
