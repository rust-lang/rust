// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DeadStoreElimination
// compile-flags: -Zmir-enable-passes=+CopyProp

#![feature(core_intrinsics)]
#![feature(custom_mir)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[inline(never)]
fn use_both(_: i32, _: i32) {}

// EMIT_MIR call_arg_copy.move_simple.DeadStoreElimination.diff
fn move_simple(x: i32) {
    use_both(x, x);
}

#[repr(packed)]
struct Packed {
    x: u8,
    y: i32,
}

// EMIT_MIR call_arg_copy.move_packed.DeadStoreElimination.diff
#[custom_mir(dialect = "analysis")]
fn move_packed(packed: Packed) {
    mir!(
        {
            Call(RET = use_both(0, packed.y), ret, UnwindContinue())
        }
        ret = {
            Return()
        }
    )
}

fn main() {
    move_simple(1);
    move_packed(Packed { x: 0, y: 1 });
}
