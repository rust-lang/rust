// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: DeadStoreElimination-final
//@ compile-flags: -Zmir-enable-passes=+CopyProp

#![feature(core_intrinsics)]
#![feature(custom_mir)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[inline(never)]
fn use_both(_: i32, _: i32) {}

// EMIT_MIR call_arg_copy.move_simple.DeadStoreElimination-final.diff
fn move_simple(x: i32) {
    // CHECK-LABEL: fn move_simple(
    // CHECK: = use_both(copy _1, move _1)
    use_both(x, x);
}

#[repr(packed)]
struct Packed {
    x: u8,
    y: i32,
}

// EMIT_MIR call_arg_copy.move_packed.DeadStoreElimination-final.diff
#[custom_mir(dialect = "analysis")]
fn move_packed(packed: Packed) {
    // CHECK-LABEL: fn move_packed(
    // CHECK: = use_both(const 0_i32, copy (_1.1: i32))
    mir! {
        {
            // We have a packed struct, verify that the copy is not turned into a move.
            Call(RET = use_both(0, packed.y), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

#[inline(never)]
fn passthrough_usize(a: usize) -> usize {
    a
}

// EMIT_MIR call_arg_copy.move_index.DeadStoreElimination-final.diff
#[custom_mir(dialect = "analysis")]
fn move_index(a: [usize; 10], b: usize) {
    // CHECK-LABEL: fn move_index(
    // CHECK: = passthrough_usize(copy _2)
    mir! {
        {
            // The index is used again after the operand is evaluated to
            // evaluate the destionation place, so the argument cannot be turned
            // into a move.
            Call(a[b] = passthrough_usize(b), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

fn main() {
    move_simple(1);
    move_packed(Packed { x: 0, y: 1 });
    move_index([0; _], 1);
}
