//@ test-mir-pass: InstSimplify-after-simplifycfg

#![feature(custom_mir, core_intrinsics)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// EMIT_MIR duplicate_switch_targets.assert_zero.InstSimplify-after-simplifycfg.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub unsafe fn assert_zero(x: u8) -> u8 {
    // CHECK-LABEL: fn assert_zero(
    // CHECK: switchInt({{.*}}) -> [0: {{bb.*}}, otherwise: {{bb.*}}]
    mir! {
        {
            match x {
                0 => retblock,
                1 => unreachable,
                _ => unreachable,
            }
        }
        unreachable = {
            Unreachable()
        }
        retblock = {
            RET = x;
            Return()
        }
    }
}
