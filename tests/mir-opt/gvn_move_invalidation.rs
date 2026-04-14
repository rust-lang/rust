// Regression test for <https://github.com/rust-lang/rust/issues/155241>:
// after `Operand::Move`-ing a non-`Copy` (`needs_drop`) place, GVN must invalidate
// the moved-from local so it cannot be reused as the destination of an
// aggregate-to-copy rewrite. Otherwise GVN may rewrite a fresh aggregate into
// `Operand::Copy(earlier_local)` and `StorageRemover` will downgrade the original
// `Move` to `Copy`, turning a single move into a double-use of freed memory.
//
//@ test-mir-pass: GVN
//@ compile-flags: -Cpanic=abort

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[inline(never)]
fn consume<T>(_: T) {}

// A struct with a non-trivial layout (more than a scalar pair, so GVN cannot
// fold it to a constant) and explicit drop glue, so `needs_drop` reports true
// and the invalidation in `simplify_operand` fires. Two `Wide { .. }` rvalues
// are still bit-identical aggregates and would receive the same `VnIndex`
// without the fix.
pub struct Wide {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}
impl Drop for Wide {
    fn drop(&mut self) {}
}
pub struct Wrap(Wide);

// EMIT_MIR gvn_move_invalidation.move_then_rebuild_droppy.GVN.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn move_then_rebuild_droppy() {
    // CHECK-LABEL: fn move_then_rebuild_droppy(
    //
    // Two distinct `Wrap(Wide { .. })` aggregates — bitwise-identical, same
    // `VnIndex` by construction — separated by a call that consumes the first.
    // GVN must NOT rewrite the second `Wrap(..)` into `_3 = copy _2` (and the
    // following `consume(move _3)` into `consume(copy _2)`): doing so would
    // make the two `move _2`/`move _3` operands two uses of `_2`'s freed
    // contents.
    //
    // CHECK: bb0: {
    // CHECK: _4 = Wide
    // CHECK: _2 = Wrap(move _4);
    // CHECK-NEXT: = consume::<Wrap>(move _2)
    //
    // CHECK: bb1: {
    // CHECK-NOT: _3 = copy _2;
    // CHECK-NOT: = consume::<Wrap>(copy _2)
    // CHECK: _5 = Wide
    // CHECK: _3 = Wrap(move _5);
    // CHECK-NEXT: = consume::<Wrap>(move _3)
    mir! {
        let _1: ();
        let _2: Wrap;
        let _3: Wrap;
        let _4: Wide;
        let _5: Wide;
        {
            _4 = Wide { a: 1_u32, b: 2_u32, c: 3_u32, d: 4_u32 };
            _2 = Wrap(Move(_4));
            Call(_1 = consume::<Wrap>(Move(_2)), ReturnTo(bb_second), UnwindUnreachable())
        }
        bb_second = {
            _5 = Wide { a: 1_u32, b: 2_u32, c: 3_u32, d: 4_u32 };
            _3 = Wrap(Move(_5));
            Call(_1 = consume::<Wrap>(Move(_3)), ReturnTo(bb_done), UnwindUnreachable())
        }
        bb_done = {
            Return()
        }
    }
}

// For comparison: when the moved type has no drop glue, the GVN unification
// still fires (post-fix) because moving a `!needs_drop` value is semantically
// equivalent to copying it and the source location is not invalidated. Here
// GVN folds the tuple to a constant and reuses the same constant for both
// calls.
//
// EMIT_MIR gvn_move_invalidation.move_then_rebuild_plain.GVN.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn move_then_rebuild_plain() {
    // CHECK-LABEL: fn move_then_rebuild_plain(
    //
    // CHECK: bb0: {
    // CHECK: _2 = const (1_u64,);
    // CHECK-NEXT: = consume::<(u64,)>(const (1_u64,))
    //
    // CHECK: bb1: {
    // GVN happily folds the second aggregate to the same constant — `(u64,)` is
    // `!needs_drop` and the post-move bit pattern is preserved.
    // CHECK: _3 = const (1_u64,);
    // CHECK-NEXT: = consume::<(u64,)>(const (1_u64,))
    mir! {
        let _1: ();
        let _2: (u64,);
        let _3: (u64,);
        {
            _2 = (1_u64,);
            Call(_1 = consume::<(u64,)>(Move(_2)), ReturnTo(bb_second), UnwindUnreachable())
        }
        bb_second = {
            _3 = (1_u64,);
            Call(_1 = consume::<(u64,)>(Move(_3)), ReturnTo(bb_done), UnwindUnreachable())
        }
        bb_done = {
            Return()
        }
    }
}

fn main() {
    move_then_rebuild_droppy();
    move_then_rebuild_plain();
}
