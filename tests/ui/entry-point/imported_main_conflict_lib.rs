// Tests that ambiguously glob importing main doesn't fail to compile in non-executable crates
// Regression test for #149412
//@ check-pass
#![crate_type = "lib"]

mod m1 { pub(crate) fn main() {} }
mod m2 { pub(crate) fn main() {} }

use m1::*;
use m2::*;
