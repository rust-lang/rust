// Tests for MaybeLiveLocals and MaybeTransitiveLiveLocals dataflows.
#![feature(core_intrinsics, custom_mir, rustc_attrs)]
#![crate_type = "lib"]
use std::intrinsics::mir::*;

// EMIT_MIR liveness.fields.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_live_locals)]
#[rustc_mir(rustc_pretty_transitive_live_locals)]
pub fn fields() -> usize {
    mir! {
        let _1: (usize, usize);
        {
            // CHECK-LABEL: fn fields(
            // CHECK:      bb0: {
            // CHECK-NEXT: live: []
            // CHECK-NEXT: live: []
            // CHECK-NEXT: _1 = (const 1_usize, const 0_usize);
            _1 = (1, 0);
            // write to _1.1 doesn't make _1 live
            // CHECK-NEXT: live: []
            // CHECK-NEXT: live: []
            // CHECK-NEXT: (_1.1: usize) = const 2_usize;
            _1.1 = 2;
            // write to _1 makes _1 dead
            // CHECK-NEXT: live: []
            // CHECK-NEXT: live: []
            // CHECK-NEXT: (const 0_usize, const 3_usize);
            _1 = (0, 3);
            // write to _1.1 doesn't make _1 dead
            // CHECK-NEXT: live: [_1]
            // CHECK-NEXT: live: [_1]
            // CHECK-NEXT: (_1.1: usize) = const 4_usize;
            _1.1 = 4;
            // read from _1.1 makes _1 live
            // CHECK-NEXT: live: [_1]
            // CHECK-NEXT: live: [_1]
            // CHECK-NEXT: _0 = copy (_1.1: usize);
            RET = _1.1;
            // CHECK-NEXT: live: [_0]
            // CHECK-NEXT: live: [_0]
            // CHECK-NEXT: return;
            Return()
        }
    }
}

// EMIT_MIR liveness.deref.dataflow.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_live_locals)]
#[rustc_mir(rustc_pretty_transitive_live_locals)]
pub fn deref() {
    mir! {
        let _1: usize;
        let _2: &mut usize;
        {
            // CHECK-LABEL: fn deref(
            // CHECK:      bb0: {
            // CHECK-NEXT: live: []
            // CHECK-NEXT: live: []
            // CHECK-NEXT: _1 = const 42_usize;
            _1 = 42;
            // CHECK-NEXT: live: [_1]
            // CHECK-NEXT: live: [_1]
            // CHECK-NEXT: _2 = &mut _1;
            _2 = &mut _1;
            // deref of _2 makes it live even on LHS of an assignment
            // CHECK-NEXT: live: [_2]
            // CHECK-NEXT: live: [_2]
            // CHECK-NEXT: (*_2) = const 24_usize;
            *_2 = 24;
            // CHECK-NEXT: live: []
            // CHECK-NEXT: live: []
            // CHECK-NEXT: _0 = ();
            RET = ();
            // CHECK-NEXT: live: [_0]
            // CHECK-NEXT: live: [_0]
            // CHECK-NEXT: return;
            Return()
        }
    }
}
