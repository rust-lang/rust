// Check that associated types print generic parameters and where clauses.
// See issue #67509.

// pretty-compare-only
// pp-exact:gat-bounds.pp

#![feature(generic_associated_types)]

trait X {
    type Y<T>: Trait where Self: Sized;
}

impl X for () {
    type Y<T> where Self: Sized = u32;
}

fn main() { }
