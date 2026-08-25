//@ edition:2015
//@ run-pass
// Regression test for issue #762


pub fn f() { }
pub fn main() { return ::f(); }
