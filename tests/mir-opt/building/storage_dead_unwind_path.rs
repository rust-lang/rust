//@ compile-flags: -Zmir-opt-level=0
// skip-filecheck
// Test that StorageDead statements are emitted on unwind paths for borrow-checking.
// This ensures the borrow-checker treats locals as dead at the same point on all paths.

// EMIT_MIR storage_dead_unwind_path.test.built.after.mir

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn test() {
    // StorageDead drop (non-drop type)
    let x = 42i32;
    // Value drop (drop type)
    let y = Droppable;
    // Function call that might panic - if it does, we should see StorageDead(x) in cleanup
    may_panic();
}

fn may_panic() {
    // This function might panic, triggering unwind
}

fn main() {
    test();
}
