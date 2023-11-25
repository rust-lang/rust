// edition:2018
// compile-flags:--extern rustc_middle
// ignore-stage1

// Test that `--extern rustc_middle` fails with `rustc_private`.

pub use rustc_middle;
//~^ ERROR use of unstable library feature 'rustc_private'

fn main() {}
