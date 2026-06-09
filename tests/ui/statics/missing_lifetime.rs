//! This test checks that we taint typeck results when there are
//! error lifetimes, even though typeck doesn't actually care about lifetimes.

struct Slice(&'reborrow [&'static [u8]]);
//~^ ERROR undeclared lifetime

static MAP: Slice = Slice(&[b"" as &'static [u8]]);

fn main() {}
