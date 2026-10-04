#![feature(core_intrinsics, custom_mir, rustc_attrs)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// EMIT_MIR precise_liveness.copied.PreciseLiveness.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn copied() -> u32 {
    // CHECK-LABEL: fn copied(
    // CHECK: [[SOURCE:_[0-9]+]] = const 42_u32;
    // CHECK-NEXT: // early: [[[SOURCE]]]
    // CHECK-NEXT: // late: [[[SOURCE]], [[DESTINATION:_[0-9]+]]]
    // CHECK-NEXT: [[DESTINATION]] = copy [[SOURCE]];
    mir! {
        let source: u32;
        let destination: u32;
        {
            source = 42;
            destination = source;
            RET = source + destination;
            Return()
        }
    }
}

// EMIT_MIR precise_liveness.moved.PreciseLiveness.0.mir
#[custom_mir(dialect = "runtime", phase = "optimized")]
#[rustc_mir(rustc_pretty_precise_liveness)]
pub fn moved() -> u32 {
    // CHECK-LABEL: fn moved(
    // CHECK: [[SOURCE:_[0-9]+]] = const 42_u32;
    // CHECK-NEXT: // early: [[[SOURCE]]]
    // CHECK-NEXT: // late: [[[DESTINATION:_[0-9]+]]]
    // CHECK-NEXT: [[DESTINATION]] = move [[SOURCE]];
    // CHECK-NEXT: // early: [[[DESTINATION]]]
    // CHECK-NEXT: // late: [_0]
    // CHECK-NEXT: _0 = move [[DESTINATION]];
    mir! {
        let source: u32;
        let destination: u32;
        {
            source = 42;
            destination = Move(source);
            RET = Move(destination);
            Return()
        }
    }
}
