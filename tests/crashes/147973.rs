// This is part of series of regression tests for some diagnostics ICEs encountered in the wild with
// suggestions having overlapping parts under https://github.com/rust-lang/rust/pull/146121.
// This is one MCVE from the beta crater run regressions from issue 147973.

//@ needs-rustc-debug-assertions
//@ known-bug: #147973

//@ aux-build: overlapping_spans_helper.rs
extern crate overlapping_spans_helper;

fn main() {
    let _name = Some(1);
    overlapping_spans_helper::do_loop!(_name);
}
