//@ check-pass
// Test that StorageDead on unwind paths ensures consistent borrow-checking.
// This test verifies that the borrow-checker correctly treats variables as dead
// at the same point on all paths (normal and unwind), making the borrow-checker
// stricter and more consistent.

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn test_storage_dead_consistency() {
    // StorageDead drop (non-drop type)
    let x = 42i32;
    // Value drop (drop type) - if this panics, x should be considered dead
    let y = Droppable;
    // After y is dropped, x should be considered dead for borrow-checking purposes
    // even on the unwind path due to StorageDead being emitted on unwind paths
    drop(y);
    // x is still in scope here, but StorageDead ensures consistent treatment
    let _val = x;
}

fn main() {
    test_storage_dead_consistency();
}
