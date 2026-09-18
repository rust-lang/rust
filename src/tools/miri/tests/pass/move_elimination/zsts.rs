//@compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]

use std::intrinsics::mir::*;

struct Zst;

// ZST return locals are allocated even without an assignment.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn uninitialized_zst() -> Zst {
    mir! {
        {
            Return()
        }
    }
}

// Moving a ZST must retain its storage and address.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn move_keeps_zst_address() -> (*const Zst, *const Zst) {
    mir! {
        let value: Zst;
        let moved: Zst;
        let before: *const Zst;
        let after: *const Zst;
        {
            value = Zst;
            before = &raw const value;
            moved = Move(value);
            after = &raw const value;
            RET = (before, after);
            Return()
        }
    }
}

fn main() {
    let _ = uninitialized_zst();
    let (before, after) = move_keeps_zst_address();
    assert_eq!(before, after);
}
