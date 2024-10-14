//! Test for cycle recover spread across two threads.
//! See `../cycles.rs` for a complete listing of cycle tests,
//! both intra and cross thread.

use crate::setup::{Knobs, ParDatabaseImpl};
use ra_salsa::ParallelDatabase;

// Recover cycle test:
//
// The pattern is as follows.
//
// Thread A                   Thread B
// --------                   --------
// a1                         b1
// |                          wait for stage 1 (blocks)
// signal stage 1             |
// wait for stage 2 (blocks)  (unblocked)
// |                          signal stage 2
// (unblocked)                wait for stage 3 (blocks)
// a2                         |
// b1 (blocks -> stage 3)     |
// |                          (unblocked)
// |                          b2
// |                          a1 (cycle detected, recovers)
// |                          b2 completes, recovers
// |                          b1 completes, recovers
// a2 sees cycle, recovers
// a1 completes, recovers

#[test]
fn parallel_cycle_all_recover() {
    let db = ParDatabaseImpl::default();
    db.knobs().signal_on_will_block.set(3);

    let thread_a = std::thread::spawn({
        let db = db.snapshot();
        move || db.a1(1)
    });

    let thread_b = std::thread::spawn({
        let db = db.snapshot();
        move || db.b1(1)
    });

    assert_eq!(thread_a.join().unwrap(), 11);
    assert_eq!(thread_b.join().unwrap(), 21);
}

#[ra_salsa::query_group(ParallelCycleAllRecover)]
pub(crate) trait TestDatabase: Knobs {
    #[ra_salsa::cycle(recover_a1)]
    fn a1(&self, key: i32) -> i32;

    #[ra_salsa::cycle(recover_a2)]
    fn a2(&self, key: i32) -> i32;

    #[ra_salsa::cycle(recover_b1)]
    fn b1(&self, key: i32) -> i32;

    #[ra_salsa::cycle(recover_b2)]
    fn b2(&self, key: i32) -> i32;
}

fn recover_a1(_db: &dyn TestDatabase, _cycle: &ra_salsa::Cycle, key: &i32) -> i32 {
    tracing::debug!("recover_a1");
    key * 10 + 1
}

fn recover_a2(_db: &dyn TestDatabase, _cycle: &ra_salsa::Cycle, key: &i32) -> i32 {
    tracing::debug!("recover_a2");
    key * 10 + 2
}

fn recover_b1(_db: &dyn TestDatabase, _cycle: &ra_salsa::Cycle, key: &i32) -> i32 {
    tracing::debug!("recover_b1");
    key * 20 + 1
}

fn recover_b2(_db: &dyn TestDatabase, _cycle: &ra_salsa::Cycle, key: &i32) -> i32 {
    tracing::debug!("recover_b2");
    key * 20 + 2
}

fn a1(db: &dyn TestDatabase, key: i32) -> i32 {
    // Wait to create the cycle until both threads have entered
    db.signal(1);
    db.wait_for(2);

    db.a2(key)
}

fn a2(db: &dyn TestDatabase, key: i32) -> i32 {
    db.b1(key)
}

fn b1(db: &dyn TestDatabase, key: i32) -> i32 {
    // Wait to create the cycle until both threads have entered
    db.wait_for(1);
    db.signal(2);

    // Wait for thread A to block on this thread
    db.wait_for(3);

    db.b2(key)
}

fn b2(db: &dyn TestDatabase, key: i32) -> i32 {
    db.a1(key)
}
