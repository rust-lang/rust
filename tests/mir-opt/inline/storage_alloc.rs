//@ test-mir-pass: Inline
//@ needs-unwind

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

unsafe extern "C-unwind" {
    fn opaque();
}

#[inline(always)]
fn callee() -> MaybeUninit<u8> {
    unsafe { opaque() };
    MaybeUninit::uninit()
}

// The destination must be allocated before the callee can unwind.
// EMIT_MIR storage_alloc.local.Inline.diff
#[custom_mir(dialect = "runtime")]
pub fn local() -> MaybeUninit<u8> {
    // CHECK-LABEL: fn local(
    // CHECK: bb0: {
    // CHECK-NEXT: StorageAlloc([[LOCAL:_[0-9]+]]);
    // CHECK: () -> [return:
    // CHECK: (cleanup): {
    // CHECK-NEXT: _0 = copy [[LOCAL]];
    mir! {
        let x: MaybeUninit<u8>;
        {
            Call(x = callee(), ReturnTo(done), UnwindCleanup(cleanup))
        }
        done = {
            RET = x;
            Return()
        }
        cleanup(cleanup) = {
            RET = x;
            UnwindResume()
        }
    }
}

// Allocating a field of the destination must also allocate its uninitialized sibling.
// EMIT_MIR storage_alloc.field.Inline.diff
#[custom_mir(dialect = "runtime")]
pub fn field() -> MaybeUninit<u8> {
    // CHECK-LABEL: fn field(
    // CHECK: bb0: {
    // CHECK-NEXT: StorageAlloc([[TUPLE:_[0-9]+]]);
    // CHECK: () -> [return:
    // CHECK: (cleanup): {
    // CHECK-NEXT: _0 = copy ([[TUPLE]].1: std::mem::MaybeUninit<u8>);
    mir! {
        let x: (MaybeUninit<u8>, MaybeUninit<u8>);
        {
            Call(x.0 = callee(), ReturnTo(done), UnwindCleanup(cleanup))
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

// StorageAlloc is needed before capturing a pointer to the destination when inlining.
// EMIT_MIR storage_alloc.indexed.Inline.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn indexed(index: usize) -> MaybeUninit<u8> {
    // CHECK-LABEL: fn indexed(
    // CHECK-SAME: [[INDEX:_[0-9]+]]: usize)
    // CHECK: StorageAlloc([[VALUES:_[0-9]+]]);
    // CHECK: = &raw mut [[VALUES]][[[INDEX]]];
    mir! {
        let values: [MaybeUninit<u8>; 2];
        {
            Call(values[index] = callee(), ReturnTo(done), UnwindContinue())
        }
        done = {
            RET = values[index];
            Return()
        }
    }
}
