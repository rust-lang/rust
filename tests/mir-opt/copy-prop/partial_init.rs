// skip-filecheck
//@ test-mir-pass: CopyProp
// Verify that we do not ICE on partial initializations.

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR partial_init.main.CopyProp.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn main() {
    mir! (
        let x: (isize, );
        {
            x.0 = 1;
            Return()
        }
    )
}
