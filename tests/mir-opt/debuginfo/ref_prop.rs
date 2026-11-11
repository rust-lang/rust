//@ test-mir-pass: ReferencePropagation
//@ compile-flags: -g -Zub_checks=false -Zinline-mir -Zmir-enable-passes=+DeadStoreElimination-initial

#![feature(core_intrinsics, custom_mir)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// EMIT_MIR ref_prop.remap_debuginfo_locals.ReferencePropagation.diff
pub fn remap_debuginfo_locals() {
    // CHECK-LABEL: fn remap_debuginfo_locals()
    // CHECK: debug a => [[a:_.*]];
    // CHECK: bb0:
    // CHECK-NEXT: [[a]] = const
    // CHECK-NEXT: DBG: [[a]] = &(*[[a]]);
    foo(&0);
}

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
#[inline]
fn foo(x: *const usize) -> &'static usize {
    mir! {
        debug a => RET;
        {
            RET = &*x;
            RET = &*x;
            Return()
        }
    }
}
