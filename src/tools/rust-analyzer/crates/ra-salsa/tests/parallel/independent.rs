use crate::setup::{ParDatabase, ParDatabaseImpl};
use ra_salsa::ParallelDatabase;

/// Test two `sum` queries (on distinct keys) executing in different
/// threads. Really just a test that `snapshot` etc compiles.
#[test]
fn in_par_two_independent_queries() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('a', 100);
    db.set_input('b', 10);
    db.set_input('c', 1);
    db.set_input('d', 200);
    db.set_input('e', 20);
    db.set_input('f', 2);

    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        move || db.sum("abc")
    });

    let thread2 = std::thread::spawn({
        let db = db.snapshot();
        move || db.sum("def")
    });

    assert_eq!(thread1.join().unwrap(), 111);
    assert_eq!(thread2.join().unwrap(), 222);
}
