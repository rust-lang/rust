//@ check-pass
//
// Regression test for issue #99286
// Tests that stabilized intrinsics are accessible
// through 'std::intrinsics', even though the module
// is unstable.

#![allow(unused_imports)]
#![allow(deprecated)]

use std::intrinsics::copy_nonoverlapping as _;
use std::intrinsics::copy as _;
use std::intrinsics::write_bytes as _;
use std::intrinsics::{copy_nonoverlapping, copy, write_bytes};

fn main() {}
