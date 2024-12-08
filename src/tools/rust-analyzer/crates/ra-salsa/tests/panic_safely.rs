use ra_salsa::{Database, ParallelDatabase, Snapshot};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};

#[ra_salsa::query_group(PanicSafelyStruct)]
trait PanicSafelyDatabase: ra_salsa::Database {
    #[ra_salsa::input]
    fn one(&self) -> usize;

    fn panic_safely(&self) -> ();

    fn outer(&self) -> ();
}

fn panic_safely(db: &dyn PanicSafelyDatabase) {
    assert_eq!(db.one(), 1);
}

static OUTER_CALLS: AtomicU32 = AtomicU32::new(0);

fn outer(db: &dyn PanicSafelyDatabase) {
    OUTER_CALLS.fetch_add(1, SeqCst);
    db.panic_safely();
}

#[ra_salsa::database(PanicSafelyStruct)]
#[derive(Default)]
struct DatabaseStruct {
    storage: ra_salsa::Storage<Self>,
}

impl ra_salsa::Database for DatabaseStruct {}

impl ra_salsa::ParallelDatabase for DatabaseStruct {
    fn snapshot(&self) -> Snapshot<Self> {
        Snapshot::new(DatabaseStruct { storage: self.storage.snapshot() })
    }
}

#[test]
fn should_panic_safely() {
    let mut db = DatabaseStruct::default();
    db.set_one(0);

    // Invoke `db.panic_safely() without having set `db.one`. `db.one` will
    // return 0 and we should catch the panic.
    let result = panic::catch_unwind(AssertUnwindSafe({
        let db = db.snapshot();
        move || db.panic_safely()
    }));
    assert!(result.is_err());

    // Set `db.one` to 1 and assert ok
    db.set_one(1);
    let result = panic::catch_unwind(AssertUnwindSafe(|| db.panic_safely()));
    assert!(result.is_ok());

    // Check, that memoized outer is not invalidated by a panic
    {
        assert_eq!(OUTER_CALLS.load(SeqCst), 0);
        db.outer();
        assert_eq!(OUTER_CALLS.load(SeqCst), 1);

        db.set_one(0);
        let result = panic::catch_unwind(AssertUnwindSafe(|| db.outer()));
        assert!(result.is_err());
        assert_eq!(OUTER_CALLS.load(SeqCst), 1);

        db.set_one(1);
        db.outer();
        assert_eq!(OUTER_CALLS.load(SeqCst), 2);
    }
}

#[test]
fn storages_are_unwind_safe() {
    fn check_unwind_safe<T: std::panic::UnwindSafe>() {}
    check_unwind_safe::<&DatabaseStruct>();
}

#[test]
fn panics_clear_query_stack() {
    let db = DatabaseStruct::default();

    // Invoke `db.panic_if_not_one() without having set `db.input`. `db.input`
    // will default to 0 and we should catch the panic.
    let result = panic::catch_unwind(AssertUnwindSafe(|| db.panic_safely()));
    assert!(result.is_err());

    // The database has been poisoned and any attempt to increment the
    // revision should panic.
    assert_eq!(db.salsa_runtime().active_query(), None);
}
