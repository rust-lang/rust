//! Test a cycle where no queries recover that occurs across threads.
//! See the `../cycles.rs` for a complete listing of cycle tests,
//! both intra and cross thread.

use crate::setup::{Knobs, ParDatabaseImpl};
use expect_test::expect;
use salsa::ParallelDatabase;

#[test]
fn parallel_cycle_none_recover() {
    let db = ParDatabaseImpl::default();
    db.knobs().signal_on_will_block.set(3);

    let thread_a = std::thread::spawn({
        let db = db.snapshot();
        move || db.a(-1)
    });

    let thread_b = std::thread::spawn({
        let db = db.snapshot();
        move || db.b(-1)
    });

    // We expect B to panic because it detects a cycle (it is the one that calls A, ultimately).
    // Right now, it panics with a string.
    let err_b = thread_b.join().unwrap_err();
    if let Some(c) = err_b.downcast_ref::<salsa::Cycle>() {
        expect![[r#"
            [
                "parallel::parallel_cycle_none_recover::AQuery::a(-1)",
                "parallel::parallel_cycle_none_recover::BQuery::b(-1)",
            ]
        "#]]
        .assert_debug_eq(&c.unexpected_participants(&db));
    } else {
        panic!("b failed in an unexpected way: {:?}", err_b);
    }

    // We expect A to propagate a panic, which causes us to use the sentinel
    // type `Canceled`.
    assert!(thread_a.join().unwrap_err().downcast_ref::<salsa::Cycle>().is_some());
}

#[salsa::query_group(ParallelCycleNoneRecover)]
pub(crate) trait TestDatabase: Knobs {
    fn a(&self, key: i32) -> i32;
    fn b(&self, key: i32) -> i32;
}

fn a(db: &dyn TestDatabase, key: i32) -> i32 {
    // Wait to create the cycle until both threads have entered
    db.signal(1);
    db.wait_for(2);

    db.b(key)
}

fn b(db: &dyn TestDatabase, key: i32) -> i32 {
    // Wait to create the cycle until both threads have entered
    db.wait_for(1);
    db.signal(2);

    // Wait for thread A to block on this thread
    db.wait_for(3);

    // Now try to execute A
    db.a(key)
}
