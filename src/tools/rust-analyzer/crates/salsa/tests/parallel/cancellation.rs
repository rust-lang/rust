use crate::setup::{CancellationFlag, Knobs, ParDatabase, ParDatabaseImpl, WithValue};
use salsa::{Cancelled, ParallelDatabase};

macro_rules! assert_cancelled {
    ($thread:expr) => {
        match $thread.join() {
            Ok(value) => panic!("expected cancellation, got {:?}", value),
            Err(payload) => match payload.downcast::<Cancelled>() {
                Ok(_) => {}
                Err(payload) => ::std::panic::resume_unwind(payload),
            },
        }
    };
}

/// Add test where a call to `sum` is cancelled by a simultaneous
/// write. Check that we recompute the result in next revision, even
/// though none of the inputs have changed.
#[test]
fn in_par_get_set_cancellation_immediate() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('a', 100);
    db.set_input('b', 10);
    db.set_input('c', 1);
    db.set_input('d', 0);

    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        move || {
            // This will not return until it sees cancellation is
            // signaled.
            db.knobs().sum_signal_on_entry.with_value(1, || {
                db.knobs()
                    .sum_wait_for_cancellation
                    .with_value(CancellationFlag::Panic, || db.sum("abc"))
            })
        }
    });

    // Wait until we have entered `sum` in the other thread.
    db.wait_for(1);

    // Try to set the input. This will signal cancellation.
    db.set_input('d', 1000);

    // This should re-compute the value (even though no input has changed).
    let thread2 = std::thread::spawn({
        let db = db.snapshot();
        move || db.sum("abc")
    });

    assert_eq!(db.sum("d"), 1000);
    assert_cancelled!(thread1);
    assert_eq!(thread2.join().unwrap(), 111);
}

/// Here, we check that `sum`'s cancellation is propagated
/// to `sum2` properly.
#[test]
fn in_par_get_set_cancellation_transitive() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('a', 100);
    db.set_input('b', 10);
    db.set_input('c', 1);
    db.set_input('d', 0);

    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        move || {
            // This will not return until it sees cancellation is
            // signaled.
            db.knobs().sum_signal_on_entry.with_value(1, || {
                db.knobs()
                    .sum_wait_for_cancellation
                    .with_value(CancellationFlag::Panic, || db.sum2("abc"))
            })
        }
    });

    // Wait until we have entered `sum` in the other thread.
    db.wait_for(1);

    // Try to set the input. This will signal cancellation.
    db.set_input('d', 1000);

    // This should re-compute the value (even though no input has changed).
    let thread2 = std::thread::spawn({
        let db = db.snapshot();
        move || db.sum2("abc")
    });

    assert_eq!(db.sum2("d"), 1000);
    assert_cancelled!(thread1);
    assert_eq!(thread2.join().unwrap(), 111);
}

/// https://github.com/salsa-rs/salsa/issues/66
#[test]
fn no_back_dating_in_cancellation() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('a', 1);
    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        move || {
            // Here we compute a long-chain of queries,
            // but the last one gets cancelled.
            db.knobs().sum_signal_on_entry.with_value(1, || {
                db.knobs()
                    .sum_wait_for_cancellation
                    .with_value(CancellationFlag::Panic, || db.sum3("a"))
            })
        }
    });

    db.wait_for(1);

    // Set unrelated input to bump revision
    db.set_input('b', 2);

    // Here we should recompuet the whole chain again, clearing the cancellation
    // state. If we get `usize::max()` here, it is a bug!
    assert_eq!(db.sum3("a"), 1);

    assert_cancelled!(thread1);

    db.set_input('a', 3);
    db.set_input('a', 4);
    assert_eq!(db.sum3("ab"), 6);
}
