//@ test-mir-pass: ScalarReplacementOfAggregates

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

// Moving a field must leave its underlying local available for the next read.
// EMIT_MIR projected_moves.repeated.ScalarReplacementOfAggregates.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn repeated() -> u32 {
    // CHECK-LABEL: fn repeated(
    // CHECK: [[FIELD:_[0-9]+]] = const 1_u32;
    // CHECK: = copy [[FIELD]];
    // CHECK-NEXT: _0 = copy [[FIELD]];
    mir! {
        let pair: (u32, u32);
        let first: u32;
        {
            pair = (1, 2);
            first = Move(pair.0);
            RET = pair.0;
            Return()
        }
    }
}
