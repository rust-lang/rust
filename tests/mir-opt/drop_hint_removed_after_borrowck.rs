//@ compile-flags: -Zmir-opt-level=0
//@ test-mir-pass: CleanupPostBorrowck
//@ skip-filecheck
// BackwardIncompatibleDropHint is only needed for borrowck/linting and should be
// removed after borrowck by CleanupPostBorrowck.

// EMIT_MIR drop_hint_removed_after_borrowck.test.built.after.mir
// EMIT_MIR drop_hint_removed_after_borrowck.test.CleanupPostBorrowck.after.mir

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn test() {
    let x = 42i32;
    let y = Droppable;
    may_panic();
}

fn may_panic() {}

fn main() {
    test();
}
