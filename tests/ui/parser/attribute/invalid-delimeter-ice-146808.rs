// regression test for #146808
//@ proc-macro: all_spans_same.rs
extern crate all_spans_same;

#[all_spans_same::all_spans_same]
#[allow{}]
fn main() {}
