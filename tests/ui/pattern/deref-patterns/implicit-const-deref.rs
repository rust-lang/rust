//! Test that we get an error about structural equality rather than a type error when attempting to
//! use const patterns of library pointer types. Currently there aren't any smart pointers that can
//! be used in constant patterns, but we still need to make sure we don't implicitly dereference the
//! scrutinee and end up with a type error; this would prevent us from reporting that only constants
//! supporting structural equality can be used as patterns.
#![feature(deref_patterns)]
#![allow(incomplete_features)]

const EMPTY: Vec<()> = Vec::new();

fn main() {
    // FIXME(inline_const_pat): if `inline_const_pat` is reinstated, there should be a case here for
    // inline const block patterns as well; they're checked differently than named constants.
    match vec![()] {
        EMPTY => {}
        //~^ ERROR: constant of non-structural type `Vec<()>` in a pattern
        _ => {}
    }
}
