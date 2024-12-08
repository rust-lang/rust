#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

// The `Drop` terminator on a type with no drop glue should be a NOP.

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn drop_in_place_with_terminator(ptr: *mut i32) {
    mir! {
        {
            Drop(*ptr, ReturnTo(after_call), UnwindContinue())
        }
        after_call = {
            Return()
        }
    }
}

pub fn main() {
    drop_in_place_with_terminator(std::ptr::without_provenance_mut(0));
    drop_in_place_with_terminator(std::ptr::without_provenance_mut(1));
}
