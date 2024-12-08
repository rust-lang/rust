use std::panic::AssertUnwindSafe;

use crate::setup::{ParDatabase, ParDatabaseImpl};
use ra_salsa::{Cancelled, ParallelDatabase};

/// Test where a read and a set are racing with one another.
/// Should be atomic.
#[test]
fn in_par_get_set_race() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('a', 100);
    db.set_input('b', 10);
    db.set_input('c', 1);

    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        move || Cancelled::catch(AssertUnwindSafe(|| db.sum("abc")))
    });

    let thread2 = std::thread::spawn(move || {
        db.set_input('a', 1000);
        db.sum("a")
    });

    // If the 1st thread runs first, you get 111, otherwise you get
    // 1011; if they run concurrently and the 1st thread observes the
    // cancellation, it'll unwind.
    let result1 = thread1.join().unwrap();
    if let Ok(value1) = result1 {
        assert!(value1 == 111 || value1 == 1011, "illegal result {value1}");
    }

    // thread2 can not observe a cancellation because it performs a
    // database write before running any other queries.
    assert_eq!(thread2.join().unwrap(), 1000);
}
