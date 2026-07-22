//@ test-mir-pass: TailCopyToMove
//@ compile-flags: -Cpanic=abort

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[derive(Copy, Clone)]
pub struct Pair {
    a: u32,
    b: u32,
}

#[derive(Copy, Clone)]
pub enum Choice {
    A(u32),
    B,
}

// EMIT_MIR tail_copy_to_move.direct.TailCopyToMove.diff
pub fn direct(x: u32) -> u32 {
    // Checks the simplest returned `Copy` local.
    // CHECK-LABEL: fn direct(
    // CHECK: _0 = move _1;
    x
}

// EMIT_MIR tail_copy_to_move.chain.TailCopyToMove.diff
pub fn chain(x: u32) -> u32 {
    // Checks that the scan propagates through a temporary local.
    // CHECK-LABEL: fn chain(
    // CHECK: [[TMP:_.*]] = move _1;
    // CHECK: _0 = move [[TMP]];
    let t = x;
    t
}

// EMIT_MIR tail_copy_to_move.aggregate.TailCopyToMove.diff
pub fn aggregate(x: u32, y: u32) -> Pair {
    // Checks aggregate construction from returned `Copy` locals.
    // CHECK-LABEL: fn aggregate(
    // CHECK: [[A:_.*]] = move _1;
    // CHECK: [[B:_.*]] = move _2;
    // CHECK: _0 = Pair { a: move [[A]], b: move [[B]] };
    Pair { a: x, b: y }
}

// EMIT_MIR tail_copy_to_move.aggregate_operands.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn aggregate_operands(x: u32, y: u32) -> (u32, u32) {
    // Checks aggregate operands that are already in the final assignment.
    // CHECK-LABEL: fn aggregate_operands(
    // CHECK: _0 = (move _1, move _2);
    mir!({
        RET = (x, y);
        Return()
    })
}

// EMIT_MIR tail_copy_to_move.projected_dest.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn projected_dest(x: u32, y: u32) -> (u32, u32) {
    // Checks assignments to direct projections of the return place.
    // CHECK-LABEL: fn projected_dest(
    // CHECK: (_0.0: u32) = move _1;
    // CHECK: (_0.1: u32) = move _2;
    mir! {
        type RET = (u32, u32);
        {
            RET.0 = x;
            RET.1 = y;
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.projected.TailCopyToMove.diff
pub fn projected(pair: Pair) -> u32 {
    // Checks that direct projected source copies are also rewritten.
    // CHECK-LABEL: fn projected(
    // CHECK: _0 = move (_1.0: u32);
    pair.a
}

// EMIT_MIR tail_copy_to_move.set_discriminant.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn set_discriminant(choice: Choice) -> Choice {
    // Checks that `SetDiscriminant` is accepted in the return tail.
    // CHECK-LABEL: fn set_discriminant(
    // CHECK: _0 = move _1;
    // CHECK: discriminant(_0) = 1;
    mir!({
        RET = choice;
        SetDiscriminant(RET, 1);
        Return()
    })
}

// EMIT_MIR tail_copy_to_move.set_discriminant_indirect.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn set_discriminant_indirect(choice: Choice) -> Choice {
    // Checks that an indirect `SetDiscriminant` place stops the scan.
    // CHECK-LABEL: fn set_discriminant_indirect(
    // CHECK: _0 = copy _1;
    // CHECK: discriminant((*[[P:_.*]])) = 1;
    mir! {
        let p: *mut Choice;

        {
            p = &raw mut choice;
            RET = choice;
            SetDiscriminant(*p, 1);
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.set_discriminant_borrowed.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn set_discriminant_borrowed(input: Choice) -> Choice {
    // Checks that writing a borrowed local's discriminant stops the scan.
    // CHECK-LABEL: fn set_discriminant_borrowed(
    // CHECK: _0 = copy _1;
    // CHECK: discriminant([[LOCAL:_.*]]) = 1;
    mir! {
        let local: Choice;
        let p: *const Choice;

        {
            p = &raw const local;
            RET = input;
            SetDiscriminant(local, 1);
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.set_discriminant_index.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn set_discriminant_index(arr: [Choice; 4], idx: usize) -> usize {
    // Checks that `SetDiscriminant` records projection locals such as indexes.
    // CHECK-LABEL: fn set_discriminant_index(
    // CHECK: _0 = copy _2;
    // CHECK: discriminant([[ARR:_.*]][_2]) = 1;
    mir! {
        let local: [Choice; 4];

        {
            local = arr;
            RET = idx;
            SetDiscriminant(local[idx], 1);
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.indirect_tail_read.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn indirect_tail_read(x: u32) -> (u32, u32) {
    // Checks that an indirect read stops the scan before earlier assignments.
    // CHECK-LABEL: fn indirect_tail_read(
    // CHECK: [[P:_.*]] = &raw const _1;
    // CHECK: [[Q:_.*]] = copy _1;
    // CHECK: [[S:_.*]] = copy (*[[P]]);
    // CHECK: _0 = (move [[Q]], move [[S]]);
    mir! {
        let p: *const u32;
        let q: u32;
        let s: u32;

        {
            p = &raw const x;
            q = x;
            s = *p;
            RET = (q, s);
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.indirect_tail_write.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn indirect_tail_write(x: u32, z: u32) -> u32 {
    // Checks that an indirect assignment destination stops the scan.
    // CHECK-LABEL: fn indirect_tail_write(
    // CHECK: _0 = copy _1;
    // CHECK: (*[[P:_.*]]) = copy _2;
    mir! {
        let p: *mut u32;

        {
            p = &raw mut x;
            RET = x;
            *p = z;
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.aggregate_with_deref.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn aggregate_with_deref(x: u32) -> (u32, u32) {
    // Checks that an indirect aggregate operand stops the aggregate scan.
    // CHECK-LABEL: fn aggregate_with_deref(
    // CHECK: [[P:_.*]] = &raw const _1;
    // CHECK: [[Q:_.*]] = copy _1;
    // CHECK: _0 = (copy [[Q]], copy (*[[P]]));
    mir! {
        let p: *const u32;
        let q: u32;

        {
            p = &raw const x;
            q = x;
            RET = (q, *p);
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.borrowed_dest_stops_tail.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn borrowed_dest_stops_tail(x: u32, z: u32) -> u32 {
    // Checks that writing to a borrowed local stops the scan.
    // CHECK-LABEL: fn borrowed_dest_stops_tail(
    // CHECK: _0 = copy _1;
    // CHECK: [[Y:_.*]] = copy _2;
    mir! {
        let y: u32;
        let p: *const u32;

        {
            p = &raw const y;
            RET = x;
            y = z;
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.unrelated_tail_store.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn unrelated_tail_store(x: u32, z: u32) -> u32 {
    // Checks that writing to an unborrowed local remains in the tail.
    // CHECK-LABEL: fn unrelated_tail_store(
    // CHECK: _0 = move _1;
    // CHECK: [[Y:_.*]] = move _2;
    mir! {
        let y: u32;

        {
            RET = x;
            y = z;
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.index_operand.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn index_operand(arr: [u32; 4], idx: usize) -> (usize, u32) {
    // Checks that index projection locals count as later uses.
    // CHECK-LABEL: fn index_operand(
    // CHECK: _0 = (copy _2, move _1[_2]);
    mir! {
        {
            RET = (idx, arr[idx]);
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.index_dest.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn index_dest(arr: [usize; 4], idx: usize) -> [usize; 4] {
    // Checks that index locals in destination projections are recorded.
    // CHECK-LABEL: fn index_dest(
    // CHECK: [[ARR:_.*]] = move _1;
    // CHECK: [[ARR]][_2] = copy _2;
    // CHECK: _0 = move [[ARR]];
    mir! {
        let a: [usize; 4];

        {
            a = arr;
            a[idx] = idx;
            RET = a;
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.repeated_operand.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn repeated_operand(x: u32) -> (u32, u32) {
    // Checks right-to-left aggregate scanning for repeated operands.
    // CHECK-LABEL: fn repeated_operand(
    // CHECK: _0 = (copy _1, move _1);
    mir!({
        RET = (x, x);
        Return()
    })
}

// EMIT_MIR tail_copy_to_move.borrowed_source_tail.TailCopyToMove.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn borrowed_source_tail(x: u32) -> u32 {
    // Checks that a borrowed source can still move at its final use.
    // CHECK-LABEL: fn borrowed_source_tail(
    // CHECK: [[P:_.*]] = &raw const _1;
    // CHECK: _0 = move _1;
    mir! {
        let p: *const u32;

        {
            p = &raw const x;
            RET = x;
            Return()
        }
    }
}

// EMIT_MIR tail_copy_to_move.shared_return.TailCopyToMove.diff
pub fn shared_return(x: u32, y: u32, take_x: bool) -> u32 {
    // Checks branch arms that share a return block.
    // CHECK-LABEL: fn shared_return(
    // CHECK: _0 = move _1;
    // CHECK: _0 = move _2;
    if take_x { x } else { y }
}
