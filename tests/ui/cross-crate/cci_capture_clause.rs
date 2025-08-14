//@ run-pass
//@ aux-build:cci_capture_clause.rs

// This test makes sure we can do cross-crate inlining on functions
// that use capture clauses.

//@ needs-threads

extern crate cci_capture_clause;

pub fn main() {
    cci_capture_clause::foo(()).recv().unwrap();
}
