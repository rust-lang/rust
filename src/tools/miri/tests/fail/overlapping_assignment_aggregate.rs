//! This is like `pass/overlapping_assignment_aggregate_scalar.rs` but with a non-scalar
//! type, and that makes it definite UB.
#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime")]
fn main() {
    mir! {
        let _1: ([u8; 1],);
        {
            _1.0 = [0_u8; 1];
            _1 = (_1.0, ); //~ERROR: overlapping ranges
            Return()
        }
    }
}
