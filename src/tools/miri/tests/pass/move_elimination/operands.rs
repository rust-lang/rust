//@revisions: normal stack tree
//@[stack]compile-flags: -Zmir-move-elimination
//@[tree]compile-flags: -Zmir-move-elimination -Zmiri-tree-borrows

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// A tail call must preserve an earlier copy before a later move frees its source.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn copy_before_move(value: [u64; 4]) -> u64 {
    mir! {
        let ptr: *const [u64; 4];
        {
            // Force the source into memory to exercise snapshotting.
            ptr = &raw const value;
            TailCall(compare(value, Move(value)))
        }
    }
}
fn compare(a: [u64; 4], b: [u64; 4]) -> u64 {
    assert_eq!(a, [42; 4]);
    assert_eq!(a, b);
    a[0]
}

// A tail call must preserve a moved field before a later move frees the whole local.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn field_before_move(value: ([u64; 4], [u64; 4])) -> u64 {
    mir! {
        {
            TailCall(compare_field(Move(value.0), Move(value)))
        }
    }
}
fn compare_field(field: [u64; 4], whole: ([u64; 4], [u64; 4])) -> u64 {
    assert_eq!(field, whole.0);
    whole.1[0]
}

// Evaluate the function pointer before a tail-call argument frees its storage.
#[derive(Clone, Copy)]
struct Callback(fn(Callback) -> u64);
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn moved_callee_storage(callback: Callback) -> u64 {
    mir! {
        let ptr: *const Callback;
        {
            // Force the function pointer into memory before evaluating the call.
            ptr = &raw const callback;
            TailCall((callback.0)(Move(callback)))
        }
    }
}
fn callback(_: Callback) -> u64 {
    42
}

// Lowering produces a CopyNonOverlapping statement. Capture the source pointer
// before moving the local containing it; a zero count avoids overlapping accesses.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn copy_zero(p: *mut u32) {
    mir! {
        let pp: *const *mut u32;
        {
            // Force the pointer local into memory to exercise snapshotting.
            pp = &raw const p;
            Call(RET = std::intrinsics::copy_nonoverlapping(p, Move(p), 0_usize), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

// Aggregate construction must preserve earlier fields before later operands free their sources.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn aggregate_before_move(value: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    mir! {
        {
            RET = (value, Move(value));
            Return()
        }
    }
}

// A repeat keeps the moved value until all elements have been written.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn repeat_move(value: [u64; 4]) -> [[u64; 4]; 2] {
    mir! {
        {
            RET = [Move(value); 2];
            Return()
        }
    }
}

// Move elimination evaluates the move before reusing the same local as the destination.
// Without it, copying an aggregate onto itself is rejected as an overlapping copy.
#[cfg(any(stack, tree))]
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn self_move(mut value: [u64; 4]) -> [u64; 4] {
    mir! {
        {
            value = Move(value);
            RET = Move(value);
            Return()
        }
    }
}

fn main() {
    assert_eq!(aggregate_before_move([1, 2, 3, 4]), ([1, 2, 3, 4], [1, 2, 3, 4]));
    assert_eq!(repeat_move([1, 2, 3, 4]), [[1, 2, 3, 4]; 2]);
    #[cfg(any(stack, tree))]
    assert_eq!(self_move([1, 2, 3, 4]), [1, 2, 3, 4]);
    assert_eq!(copy_before_move([42; 4]), 42);
    assert_eq!(field_before_move(([1; 4], [2; 4])), 2);
    assert_eq!(moved_callee_storage(Callback(callback)), 42);
    let mut value = 42;
    copy_zero(&raw mut value);
    assert_eq!(value, 42);
}
