//! Ensure that the MIR validator runs on Miri's input.
//@normalize-stderr-test: "\n +[0-9]+:[^\n]+" -> ""
//@normalize-stderr-test: "\n +at [^\n]+" -> ""
//@normalize-stderr-test: "\n +\[\.\.\. omitted [0-9]+ frames? \.\.\.\]" -> ""
//@normalize-stderr-test: "\n[ =]*note:.*" -> ""
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
