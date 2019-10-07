// ignore-stage1
// edition:2018
// compile-flags:--extern rustc

// Test that `--extern rustc` fails with `rustc_private`.

pub use rustc;
//~^ ERROR use of unstable library feature 'rustc_private'

fn main() {}
