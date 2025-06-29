// skip-filecheck
//@ test-mir-pass: CopyProp

#![feature(custom_mir, core_intrinsics)]

// Check that we do not remove the storage statements if the head
// is uninitialized in an unreachable block.

use std::intrinsics::mir::*;

// EMIT_MIR copy_prop_storage_unreachable.f.CopyProp.diff

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn f(_1: &mut usize) {
    mir! {
        let _2: usize;
        let _3: usize;
        {
            StorageLive(_2);
            _2 = 42;
            _3 = _2;
            (*_1) = _3;
            StorageDead(_2);
            Return()
        }
        bb1 = {
            // Ensure that _2 is considered uninitialized by `MaybeUninitializedLocals`.
            StorageLive(_2);
            // Use of _3 (in an unreachable block) when definition of _2 is unavailable.
            (*_1) = _3;
            StorageDead(_2);
            Return()
        }
    }
}
