//! Test for cycle recover spread across two threads.
//! See `../cycles.rs` for a complete listing of cycle tests,
//! both intra and cross thread.

use crate::setup::{Knobs, ParDatabaseImpl};
use salsa::ParallelDatabase;

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
// |                          |
// |                          b2
// |                          b3
// |                          a1 (blocks -> stage 2)
// (unblocked)                |
// a2 (cycle detected)        |
//                            b3 recovers
//                            b2 resumes
//                            b1 panics because bug

#[test]
fn parallel_cycle_mid_recovers() {
    let db = ParDatabaseImpl::default();
    db.knobs().signal_on_will_block.set(2);

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

#[salsa::query_group(ParallelCycleMidRecovers)]
pub(crate) trait TestDatabase: Knobs {
    fn a1(&self, key: i32) -> i32;

    fn a2(&self, key: i32) -> i32;

    #[salsa::cycle(recover_b1)]
    fn b1(&self, key: i32) -> i32;

    fn b2(&self, key: i32) -> i32;

    #[salsa::cycle(recover_b3)]
    fn b3(&self, key: i32) -> i32;
}

fn recover_b1(_db: &dyn TestDatabase, _cycle: &salsa::Cycle, key: &i32) -> i32 {
    tracing::debug!("recover_b1");
    key * 20 + 2
}

fn recover_b3(_db: &dyn TestDatabase, _cycle: &salsa::Cycle, key: &i32) -> i32 {
    tracing::debug!("recover_b1");
    key * 200 + 2
}

fn a1(db: &dyn TestDatabase, key: i32) -> i32 {
    // tell thread b we have started
    db.signal(1);

    // wait for thread b to block on a1
    db.wait_for(2);

    db.a2(key)
}

fn a2(db: &dyn TestDatabase, key: i32) -> i32 {
    // create the cycle
    db.b1(key)
}

fn b1(db: &dyn TestDatabase, key: i32) -> i32 {
    // wait for thread a to have started
    db.wait_for(1);

    db.b2(key);

    0
}

fn b2(db: &dyn TestDatabase, key: i32) -> i32 {
    // will encounter a cycle but recover
    db.b3(key);
    db.b1(key); // hasn't recovered yet
    0
}

fn b3(db: &dyn TestDatabase, key: i32) -> i32 {
    // will block on thread a, signaling stage 2
    db.a1(key)
}
