//! Ensure that the MIR validator runs on Miri's input.
//@rustc-env:RUSTC_ICE=0
//@normalize-stderr-test: "\n +[0-9]+:.+" -> ""
//@normalize-stderr-test: "\n +at .+" -> ""
//@normalize-stderr-test: "\n +\[\.\.\. omitted [0-9]+ frames? \.\.\.\].*" -> ""
//@normalize-stderr-test: "\n[ =]*note:.*" -> ""
//@normalize-stderr-test: "DefId\([^()]*\)" -> "DefId"
// Somehow on rustc Windows CI, the "Miri caused an ICE" message is not shown
// and we don't even get a regular panic; rustc aborts with a different exit code instead.
//@ignore-host: windows

// FIXME: this tests a crash in rustc. For stage1, rustc is built with the downloaded standard
// library which doesn't yet print the thread ID. Normalization can be removed at the stage bump.
// For the grep: cfg(bootstrap)
//@normalize-stderr-test: "thread 'rustc' panicked" -> "thread 'rustc' ($$TID) panicked"

#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn main() {
    mir! {
        let x: i32;
        let tuple: (*mut i32,);
        {
            tuple.0 = core::ptr::addr_of_mut!(x);
            // Deref at the wrong place!
            *(tuple.0) = 1;
            Return()
        }
    }
}
