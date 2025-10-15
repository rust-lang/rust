//@ test-mir-pass: DestinationPropagation
//@ compile-flags: -g -Zmir-enable-passes=+DeadStoreElimination-initial

#![feature(core_intrinsics, custom_mir)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// EMIT_MIR dest_prop.remap_debuginfo_locals.DestinationPropagation.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn remap_debuginfo_locals(a: bool, b: &bool) -> &bool {
    // CHECK-LABEL: fn remap_debuginfo_locals(
    // CHECK: debug c => [[c:_.*]];
    // CHECK: bb0:
    // CHECK-NEXT: DBG: [[c]] = &_1;
    mir! {
        let _3: &bool;
        let _4: bool;
        debug c => _3;
        {
            _3 = &a;
            StorageLive(_4);
            _4 = a;
            _3 = b;
            match _4 {
                true => bb1,
                _ => bb2,
            }
        }
        bb1 = {
            Goto(bb2)
        }
        bb2 = {
            StorageDead(_4);
            RET = _3;
            Return()
        }
    }
}
