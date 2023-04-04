#![feature(custom_mir, core_intrinsics)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// unit-test: InstCombine

// EMIT_MIR instcombine_duplicate_switch_targets.assert_zero.InstCombine.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub unsafe fn assert_zero(x: u8) -> u8 {
    mir!(
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
    )
}
