// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: DeadStoreElimination-final
//@ compile-flags: -Zmir-enable-passes=+CopyProp

#![feature(core_intrinsics)]
#![feature(custom_mir)]
#![allow(internal_features)]

use std::convert::identity;
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

// EMIT_MIR call_arg_copy.move_index.DeadStoreElimination-final.diff
#[custom_mir(dialect = "analysis")]
fn move_index(a: [usize; 10], b: usize) {
    // CHECK-LABEL: fn move_index(
    // CHECK: = identity::<usize>(copy _2)
    mir! {
        {
            // The index is used again after the operand is evaluated to
            // evaluate the destination place, so the argument cannot be turned
            // into a move.
            Call(a[b] = identity(b), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

// EMIT_MIR call_arg_copy.ret_is_arg.DeadStoreElimination-final.diff
#[custom_mir(dialect = "runtime")]
fn ret_is_arg(x: [u64; 5]) -> [u64; 5] {
    // CHECK-LABEL: fn ret_is_arg(_1
    // CHECK: _1 = identity::<[u64; 5]>(copy _1) ->
    mir! {
        {
            Call(x = identity(x), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET = x;
            Return()
        }
    }
}

fn main() {
    move_simple(1);
    move_packed(Packed { x: 0, y: 1 });
    move_index([0; _], 1);
    ret_is_arg([0; 5]);
}
