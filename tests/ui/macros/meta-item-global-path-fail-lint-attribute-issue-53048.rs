// Regression test for https://github.com/rust-lang/rust/issues/53048/
// This test ensures that invalid meta item or global path
// is rejected by the compiler.

// value must be a literal, not a path
#[allow(a = ::b::c)]
//~^ ERROR expected unsuffixed literal, found `::`

fn main() {}
