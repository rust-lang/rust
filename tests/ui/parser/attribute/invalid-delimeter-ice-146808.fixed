// regression test for #146808
//@ proc-macro: all_spans_same.rs
//@ run-rustfix
// the fixed program is still broken, but rustfix didn't crash!
// that's what we want to test here.
//@ rustfix-dont-test-fixed

extern crate all_spans_same;

#[all_spans_same::all_spans_same]
//~^ ERROR wrong meta list delimiters
#[allow{}]
fn main() {}
