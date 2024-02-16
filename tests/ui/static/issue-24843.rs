//@ aux-build: issue_24843.rs
//@ check-pass

extern crate issue_24843;

static _TEST_STR_2: &'static str = &issue_24843::TEST_STR;

fn main() {}
