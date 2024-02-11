//! Test for cycle recover spread across two threads.
//! See `../cycles.rs` for a complete listing of cycle tests,
//! both intra and cross thread.

use crate::setup::{Knobs, ParDatabaseImpl};
use salsa::ParallelDatabase;
use test_log::test;

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
// |                          a1 (cycle detected)
// a2 recovery fn executes    |
// a1 completes normally      |
//                            b2 completes, recovers
//                            b1 completes, recovers

#[test]
fn parallel_cycle_one_recovers() {
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

    // We expect that the recovery function yields
    // `1 * 20 + 2`, which is returned (and forwarded)
    // to b1, and from there to a2 and a1.
    assert_eq!(thread_a.join().unwrap(), 22);
    assert_eq!(thread_b.join().unwrap(), 22);
}

#[salsa::query_group(ParallelCycleOneRecovers)]
pub(crate) trait TestDatabase: Knobs {
    fn a1(&self, key: i32) -> i32;

    #[salsa::cycle(recover)]
    fn a2(&self, key: i32) -> i32;

    fn b1(&self, key: i32) -> i32;

    fn b2(&self, key: i32) -> i32;
}

fn recover(_db: &dyn TestDatabase, _cycle: &salsa::Cycle, key: &i32) -> i32 {
    tracing::debug!("recover");
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
