//! Regression test for https://github.com/rust-lang/rust/issues/13446

// Used to cause ICE

static VEC: [u32; 256] = vec![];
//~^ ERROR mismatched types

fn main() {}
