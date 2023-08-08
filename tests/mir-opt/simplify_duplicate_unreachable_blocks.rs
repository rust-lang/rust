#![feature(custom_mir, core_intrinsics)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// unit-test: SimplifyCfg-after-uninhabited-enum-branching

// EMIT_MIR simplify_duplicate_unreachable_blocks.assert_nonzero_nonmax.SimplifyCfg-after-uninhabited-enum-branching.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub unsafe fn assert_nonzero_nonmax(x: u8) -> u8 {
    mir!(
        {
            match x {
                0 => unreachable1,
                u8::MAX => unreachable2,
                _ => retblock,
            }
        }
        unreachable1 = {
            Unreachable()
        }
        unreachable2 = {
            Unreachable()
        }
        retblock = {
            RET = x;
            Return()
        }
    )
}
