//@ test-mir-pass: ScalarReplacementOfAggregates
//@ needs-unwind

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

// A field write must also allocate the uninitialized sibling field after SROA.
// EMIT_MIR partial_write.assignment.ScalarReplacementOfAggregates.diff
#[custom_mir(dialect = "runtime")]
pub fn assignment() -> MaybeUninit<usize> {
    // CHECK-LABEL: fn assignment(
    // CHECK: bb0: {
    // CHECK-NEXT: StorageAlloc([[SIBLING:_[0-9]+]]);
    // CHECK-NEXT: {{_[0-9]+}} = const 0_usize;
    // CHECK-NEXT: _0 = copy [[SIBLING]];
    mir! {
        let x: (usize, MaybeUninit<usize>);
        {
            x.0 = 0;
            RET = x.1;
            Return()
        }
    }
}

// Call destinations must allocate sibling fields before the call.
// EMIT_MIR partial_write.call.ScalarReplacementOfAggregates.diff
#[custom_mir(dialect = "runtime")]
pub fn call(f: fn() -> usize) -> MaybeUninit<usize> {
    // CHECK-LABEL: fn call(
    // CHECK: bb0: {
    // CHECK-NEXT: StorageAlloc([[CALL_SIBLING:_[0-9]+]]);
    // CHECK-NEXT: {{_[0-9]+}} = copy _1() ->
    // CHECK: _0 = copy [[CALL_SIBLING]];
    // CHECK: (cleanup): {
    // CHECK-NEXT: _0 = copy [[CALL_SIBLING]];
    mir! {
        let x: (usize, MaybeUninit<usize>);
        {
            Call(x.0 = f(), ReturnTo(done), UnwindCleanup(cleanup))
        }
        done = {
            RET = x.1;
            Return()
        }
        cleanup(cleanup) = {
            RET = x.1;
            UnwindResume()
        }
    }
}

fn main() {}
