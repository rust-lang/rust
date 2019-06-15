// FIXME: Individual imports of built-in macros are not stability checked right now,
// so the whole feature is gated instead.

// edition:2018
// gate-test-builtin_macro_imports

use concat as stable; //~ ERROR imports of built-in macros are unstable
use concat_idents as unstable; //~ ERROR imports of built-in macros are unstable

fn main() {}
