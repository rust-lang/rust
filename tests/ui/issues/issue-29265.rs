//@ aux-build:issue-29265.rs
//@ check-pass

extern crate issue_29265 as lib;

static _UNUSED: &'static lib::SomeType = &lib::SOME_VALUE;

fn main() {
    vec![0u8; lib::SOME_VALUE.some_member];
}
