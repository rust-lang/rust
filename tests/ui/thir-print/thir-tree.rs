//@ check-pass
//@ compile-flags: -Z unpretty=thir-tree
//@ revisions: non_incr incr
//@[incr] incremental
//
// We do incremental to see the span parents printed. (Span parents are unused in non-incremental.)

pub fn main() {}
