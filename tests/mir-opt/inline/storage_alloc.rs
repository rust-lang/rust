//@ test-mir-pass: Inline

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

// StorageAlloc is needed before capturing a pointer to the destination when
// inlining.
// EMIT_MIR storage_alloc.indexed.Inline.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn indexed(index: usize) -> u32 {
    // CHECK-LABEL: fn indexed(
    // CHECK-SAME: [[INDEX:_[0-9]+]]: usize)
    // CHECK: StorageAlloc([[VALUES:_[0-9]+]]);
    // CHECK: = &raw mut [[VALUES]][[[INDEX]]];
    mir! {
        let values: [u32; 2];
        {
            Call(values[index] = identity(42), ReturnTo(done), UnwindUnreachable())
        }
        done = {
            RET = values[index];
            Return()
        }
    }
}

#[inline(always)]
fn identity(value: u32) -> u32 {
    value
}
