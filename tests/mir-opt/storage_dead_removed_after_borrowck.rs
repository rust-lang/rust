//@ compile-flags: -Zmir-opt-level=0
//@ test-mir-pass: CleanupPostBorrowck
// skip-filecheck
// Test that StorageDead statements are removed after borrow-checking by CleanupPostBorrowck.
// StorageDead is only needed for borrow-checking and should be removed before optimization passes.

// EMIT_MIR storage_dead_removed_after_borrowck.test.built.after.mir
// EMIT_MIR storage_dead_removed_after_borrowck.test.CleanupPostBorrowck.after.mir

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn test() {
    // StorageDead drop (non-drop type)
    let x = 42i32;
    // Value drop (drop type)
    let y = Droppable;
    // Function call that might panic
    may_panic();
}

fn may_panic() {
    // This function might panic, triggering unwind
}

fn main() {
    test();
}
