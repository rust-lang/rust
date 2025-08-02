//@ test-mir-pass: RemoveUnneededDrops

#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

// EMIT_MIR remove_unneeded_drops.opt.RemoveUnneededDrops.diff
#[custom_mir(dialect = "runtime")]
fn opt(x: bool) {
    // CHECK-LABEL: fn opt(
    // CHECK-NOT: drop(
    mir! {
        { Drop(x, ReturnTo(bb1), UnwindUnreachable()) }
        bb1 = { Return() }
    }
}

// EMIT_MIR remove_unneeded_drops.dont_opt.RemoveUnneededDrops.diff
#[custom_mir(dialect = "runtime")]
fn dont_opt(x: Vec<bool>) {
    // CHECK-LABEL: fn dont_opt(
    // CHECK: drop(
    mir! {
        { Drop(x, ReturnTo(bb1), UnwindUnreachable()) }
        bb1 = { Return() }
    }
}

// EMIT_MIR remove_unneeded_drops.opt_generic_copy.RemoveUnneededDrops.diff
#[custom_mir(dialect = "runtime")]
fn opt_generic_copy<T: Copy>(x: T) {
    // CHECK-LABEL: fn opt_generic_copy(
    // CHECK-NOT: drop(
    mir! {
        { Drop(x, ReturnTo(bb1), UnwindUnreachable()) }
        bb1 = { Return() }
    }
}

// EMIT_MIR remove_unneeded_drops.cannot_opt_generic.RemoveUnneededDrops.diff
// since the pass is not running on monomorphisized code,
// we can't (but probably should) optimize this
#[custom_mir(dialect = "runtime")]
fn cannot_opt_generic<T>(x: T) {
    // CHECK-LABEL: fn cannot_opt_generic(
    // CHECK: drop(
    mir! {
        { Drop(x, ReturnTo(bb1), UnwindUnreachable()) }
        bb1 = { Return() }
    }
}

fn main() {
    // CHECK-LABEL: fn main(
    opt(true);
    opt_generic_copy(42);
    cannot_opt_generic(42);
    dont_opt(vec![true]);
}
