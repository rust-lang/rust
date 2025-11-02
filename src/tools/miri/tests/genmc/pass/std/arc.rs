//@compile-flags: -Zmiri-genmc -Zmiri-disable-stacked-borrows
//@revisions: check_count try_upgrade
//@normalize-stderr-test: "\n *= note: inside `std::.*" -> ""

// Check that various operations on `std::sync::Arc` are handled properly in GenMC mode.
//
// The number of explored executions in the expected output of this test may change if
// the implementation of `Arc` is ever changed, or additional optimizations are added to GenMC mode.
//
// The revision that tries to upgrade the `Weak` should never explore fewer executions compared to the revision that just accesses the `strong_count`,
// since `upgrade` needs to access the `strong_count` internally.
// There should always be more than 1 execution for both, since there is always the possibilility that the `Arc` has already been dropped, or it hasn't.

use std::sync::Arc;

fn main() {
    let data = Arc::new(42);

    // Clone the Arc, drop the original, check that memory still valid.
    let data_clone = Arc::clone(&data);
    drop(data);
    assert!(*data_clone == 42);

    // Create a Weak reference.
    let weak = Arc::downgrade(&data_clone);

    // Spawn a thread that uses the Arc.
    let weak_ = weak.clone();
    let handle = std::thread::spawn(move || {
        // Try to upgrade weak reference.
        // Depending on execution schedule, this may fail or succeed depending on whether this runs before or after the `drop` in the main thread.

        #[cfg(check_count)]
        let _strong_count = weak_.strong_count();

        #[cfg(try_upgrade)]
        if let Some(strong) = weak_.upgrade() {
            assert_eq!(*strong, 42);
        }
    });

    // Drop the last strong reference to the data.
    drop(data_clone);

    // Wait for the thread to finish.
    handle.join().unwrap();

    // The upgrade should fail now (all Arcs dropped).
    assert!(weak.upgrade().is_none());
}
