use crate::setup::{ParDatabase, ParDatabaseImpl};
use crate::signal::Signal;
use ra_salsa::{Database, ParallelDatabase};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

/// Add test where a call to `sum` is cancelled by a simultaneous
/// write. Check that we recompute the result in next revision, even
/// though none of the inputs have changed.
#[test]
fn in_par_get_set_cancellation() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('a', 1);

    let signal = Arc::new(Signal::default());

    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        let signal = signal.clone();
        move || {
            // Check that cancellation flag is not yet set, because
            // `set` cannot have been called yet.
            catch_unwind(AssertUnwindSafe(|| db.unwind_if_cancelled())).unwrap();

            // Signal other thread to proceed.
            signal.signal(1);

            // Wait for other thread to signal cancellation
            catch_unwind(AssertUnwindSafe(|| loop {
                db.unwind_if_cancelled();
                std::thread::yield_now();
            }))
            .unwrap_err();
        }
    });

    let thread2 = std::thread::spawn({
        move || {
            // Wait until thread 1 has asserted that they are not cancelled
            // before we invoke `set.`
            signal.wait_for(1);

            // This will block until thread1 drops the revision lock.
            db.set_input('a', 2);

            db.input('a')
        }
    });

    thread1.join().unwrap();

    let c = thread2.join().unwrap();
    assert_eq!(c, 2);
}
