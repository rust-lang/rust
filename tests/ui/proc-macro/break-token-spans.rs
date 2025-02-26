//@ proc-macro: test-macros.rs
// Regression test for issues #68489 and #70987
// Tests that we properly break tokens in `probably_equal_for_proc_macro`
// See #72306
//
// Note that the weird spacing in this example is critical
// for testing the issue.

extern crate test_macros;

#[test_macros::recollect_attr]
fn repro() {
    f :: < Vec < _ > > ( ) ; //~ ERROR cannot find
    let a: Option<Option<u8>>= true; //~ ERROR mismatched
}
fn main() {}
