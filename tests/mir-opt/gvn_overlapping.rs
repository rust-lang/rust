//@ test-mir-pass: GVN

#![feature(custom_mir, core_intrinsics)]

// Check that we do not create overlapping assignments.

use std::intrinsics::mir::*;

// EMIT_MIR gvn_overlapping.overlapping.GVN.diff
#[custom_mir(dialect = "runtime")]
fn overlapping(_17: Adt) {
    // CHECK-LABEL: fn overlapping(
    // CHECK: let mut [[PTR:.*]]: *mut Adt;
    // CHECK: (*[[PTR]]) = Adt::Some(copy {{.*}});
    mir! {
        let _33: *mut Adt;
        let _48: u32;
        let _73: &Adt;
        {
            _33 = core::ptr::addr_of_mut!(_17);
            _73 = &(*_33);
            _48 = Field(Variant((*_73), 1), 0);
            (*_33) = Adt::Some(_48);
            Return()
        }
    }
}

fn main() {
    overlapping(Adt::Some(0));
}

enum Adt {
    None,
    Some(u32),
}
