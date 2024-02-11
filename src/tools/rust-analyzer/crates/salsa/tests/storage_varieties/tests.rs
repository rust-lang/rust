#![cfg(test)]

use crate::implementation::DatabaseImpl;
use crate::queries::Database;
use salsa::Database as _Database;
use salsa::Durability;

#[test]
fn memoized_twice() {
    let db = DatabaseImpl::default();
    let v1 = db.memoized();
    let v2 = db.memoized();
    assert_eq!(v1, v2);
}

#[test]
fn volatile_twice() {
    let mut db = DatabaseImpl::default();
    let v1 = db.volatile();
    let v2 = db.volatile(); // volatiles are cached, so 2nd read returns the same
    assert_eq!(v1, v2);

    db.salsa_runtime_mut().synthetic_write(Durability::LOW); // clears volatile caches

    let v3 = db.volatile(); // will re-increment the counter
    let v4 = db.volatile(); // second call will be cached
    assert_eq!(v1 + 1, v3);
    assert_eq!(v3, v4);
}

#[test]
fn intermingled() {
    let mut db = DatabaseImpl::default();
    let v1 = db.volatile();
    let v2 = db.memoized();
    let v3 = db.volatile(); // cached
    let v4 = db.memoized(); // cached

    assert_eq!(v1, v2);
    assert_eq!(v1, v3);
    assert_eq!(v2, v4);

    db.salsa_runtime_mut().synthetic_write(Durability::LOW); // clears volatile caches

    let v5 = db.memoized(); // re-executes volatile, caches new result
    let v6 = db.memoized(); // re-use cached result
    assert_eq!(v4 + 1, v5);
    assert_eq!(v5, v6);
}
