#![feature(core_intrinsics, custom_mir)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// EMIT_MIR dead_on_invalid_place.invalid_place.PreCodegen.after.mir
#[custom_mir(dialect = "runtime")]
pub fn invalid_place(c: bool) -> bool {
    // CHECK-LABEL: fn invalid_place
    // CHECK: debug c1_ref => [[c1_ref:_[0-9]+]];
    // CHECK: bb0: {
    // We cannot read the reference, since `c1` is dead.
    // CHECK-NEXT: DBG: [[c1_ref]] = &?
    // CHECK-NEXT: _0 = copy _1;
    // CHECK-NEXT: return;
    mir! {
        let _c1_ref: &bool;
        let c1: bool;
        debug c1_ref => _c1_ref;
        {
            c1 = c;
            _c1_ref = &c1;
            RET = c;
            Return()
        }
    }
}
