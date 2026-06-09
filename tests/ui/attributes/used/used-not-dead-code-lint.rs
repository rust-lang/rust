//! Checks that the `dead_code` lint does not consider `#[used]` items unused.
//! Regression test for <https://github.com/rust-lang/rust/issues/41628>.

//@ check-pass
#![deny(dead_code)]

#[used]
static FOO: u32 = 0;

fn main() {}
