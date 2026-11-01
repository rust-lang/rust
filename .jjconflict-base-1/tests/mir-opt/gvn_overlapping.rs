//@ test-mir-pass: GVN

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

// EMIT_MIR gvn_overlapping.overlapping.GVN.diff
/// Check that we do not create overlapping assignments.
#[custom_mir(dialect = "runtime")]
fn overlapping(_1: Adt) {
    // CHECK-LABEL: fn overlapping(
    // CHECK: let mut [[PTR:.*]]: &mut Adt;
    // CHECK: (*[[PTR]]) = Adt::Some(copy {{.*}});
    mir! {
        let _2: &mut Adt;
        let _3: u32;
        let _4: &Adt;
        {
            _2 = &mut _1;
            _4 = &(*_2);
            _3 = Field(Variant((*_4), 1), 0);
            (*_2) = Adt::Some(_3);
            Return()
        }
    }
}

// EMIT_MIR gvn_overlapping.stable_projection.GVN.diff
/// Check that we allow dereferences in the RHS if the LHS is a stable projection.
#[custom_mir(dialect = "runtime")]
fn stable_projection(_1: (Adt,)) {
    // CHECK-LABEL: fn stable_projection(
    // CHECK: let mut _2: &Adt;
    // CHECK: let mut _4: &Adt;
    // CHECK: (_5.0: Adt) = copy (_1.0: Adt);
    mir! {
        let _2: &Adt;
        let _3: u32;
        let _4: &Adt;
        let _5: (Adt,);
        {
            _2 = &_1.0;
            _4 = &(*_2);
            _3 = Field(Variant((*_4), 1), 0);
            _5.0 = Adt::Some(_3);
            Return()
        }
    }
}

// EMIT_MIR gvn_overlapping.fields.GVN.diff
/// Check that we do not create assignments between different fields of the same local.
#[custom_mir(dialect = "runtime")]
fn fields(_1: (Adt, Adt)) {
    // CHECK-LABEL: fn fields(
    // CHECK: _2 = copy (((_1.0: Adt) as variant#1).0: u32);
    // CHECK-NEXT: (_1.1: Adt) = Adt::Some(copy _2);
    mir! {
        let _2: u32;
        {
            _2 = Field(Variant(_1.0, 1), 0);
            _1.1 = Adt::Some(_2);
            Return()
        }
    }
}

// EMIT_MIR gvn_overlapping.copy_overlapping.GVN.diff
#[custom_mir(dialect = "runtime")]
fn copy_overlapping() {
    mir! {
        let _1;
        let _2;
        let _3;
        {
            place!(Field(Variant(_1, 1), 0)) = 0u32;
            _3 = &_1;
            _2 = Field(Variant(*_3, 1), 0);
            _1 = Adt::Some(_2);
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
