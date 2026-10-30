//@ test-mir-pass: RemoveZsts

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

// Under move elimination semantics we can't remove direct assignments since they have
// the side effect of creating an allocation for a local. If they are truly
// dead then they will be eliminated later optimization passes.

// Preserve ZST writes while removing storage markers and replacing operands with constants.
// EMIT_MIR remove_zsts_move_elimination.local.RemoveZsts.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn local() {
    // CHECK-LABEL: fn local(
    // CHECK-NOT: StorageLive
    // CHECK: {{_[0-9]+}} = ();
    // CHECK-NEXT: _0 = const ();
    // CHECK-NOT: StorageDead
    // CHECK: return;
    mir! {
        let value: ();
        {
            StorageLive(value);
            value = ();
            RET = Move(value);
            StorageDead(value);
            Return()
        }
    }
}

pub enum Unit {
    Value,
}

// Preserve ZST SetDiscriminant since it allocates the destination local.
// EMIT_MIR remove_zsts_move_elimination.discriminant.RemoveZsts.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn discriminant() -> Unit {
    // CHECK-LABEL: fn discriminant(
    // CHECK: discriminant(_0) = 0;
    // CHECK-NEXT: return;
    mir! {
        {
            SetDiscriminant(RET, 0);
            Return()
        }
    }
}

// Indirect ZST writes cannot allocate a local and can still be removed.
// EMIT_MIR remove_zsts_move_elimination.indirect.RemoveZsts.diff
#[custom_mir(dialect = "runtime", phase = "initial")]
pub unsafe fn indirect(value: *mut (), discriminant: *mut Unit) {
    // CHECK-LABEL: fn indirect(
    // CHECK-NOT: (*
    // CHECK: _0 = ();
    // CHECK-NEXT: return;
    mir! {
        {
            *value = ();
            SetDiscriminant(*discriminant, 0);
            RET = ();
            Return()
        }
    }
}
