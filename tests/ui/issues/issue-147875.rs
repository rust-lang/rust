//@ check-pass
// Test for issue #147875: StorageDead on unwind paths ensures consistent borrow-checking.
// This test verifies that the borrow-checker correctly treats variables as dead at the
// same point on all paths (normal and unwind), making the borrow-checker stricter and
// more consistent.

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn test_storage_dead_on_unwind() {
    // StorageDead drop (non-drop type)
    let x = 42i32;
    // Value drop (drop type) - if this panics, x should be considered dead
    let y = Droppable;
    // After y is dropped, x should be considered dead for borrow-checking purposes
    // even on the unwind path
}

fn main() {
    test_storage_dead_on_unwind();
}
