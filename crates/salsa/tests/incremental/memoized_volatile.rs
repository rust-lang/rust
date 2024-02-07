use crate::implementation::{TestContext, TestContextImpl};
use salsa::{Database, Durability};

#[salsa::query_group(MemoizedVolatile)]
pub(crate) trait MemoizedVolatileContext: TestContext {
    // Queries for testing a "volatile" value wrapped by
    // memoization.
    fn memoized2(&self) -> usize;
    fn memoized1(&self) -> usize;
    fn volatile(&self) -> usize;
}

fn memoized2(db: &dyn MemoizedVolatileContext) -> usize {
    db.log().add("Memoized2 invoked");
    db.memoized1()
}

fn memoized1(db: &dyn MemoizedVolatileContext) -> usize {
    db.log().add("Memoized1 invoked");
    let v = db.volatile();
    v / 2
}

fn volatile(db: &dyn MemoizedVolatileContext) -> usize {
    db.log().add("Volatile invoked");
    db.salsa_runtime().report_untracked_read();
    db.clock().increment()
}

#[test]
fn volatile_x2() {
    let query = TestContextImpl::default();

    // Invoking volatile twice doesn't execute twice, because volatile
    // queries are memoized by default.
    query.volatile();
    query.volatile();
    query.assert_log(&["Volatile invoked"]);
}

/// Test that:
///
/// - On the first run of R0, we recompute everything.
/// - On the second run of R1, we recompute nothing.
/// - On the first run of R1, we recompute Memoized1 but not Memoized2 (since Memoized1 result
///   did not change).
/// - On the second run of R1, we recompute nothing.
/// - On the first run of R2, we recompute everything (since Memoized1 result *did* change).
#[test]
fn revalidate() {
    let mut query = TestContextImpl::default();

    query.memoized2();
    query.assert_log(&["Memoized2 invoked", "Memoized1 invoked", "Volatile invoked"]);

    query.memoized2();
    query.assert_log(&[]);

    // Second generation: volatile will change (to 1) but memoized1
    // will not (still 0, as 1/2 = 0)
    query.salsa_runtime_mut().synthetic_write(Durability::LOW);
    query.memoized2();
    query.assert_log(&["Volatile invoked", "Memoized1 invoked"]);
    query.memoized2();
    query.assert_log(&[]);

    // Third generation: volatile will change (to 2) and memoized1
    // will too (to 1).  Therefore, after validating that Memoized1
    // changed, we now invoke Memoized2.
    query.salsa_runtime_mut().synthetic_write(Durability::LOW);

    query.memoized2();
    query.assert_log(&["Volatile invoked", "Memoized1 invoked", "Memoized2 invoked"]);

    query.memoized2();
    query.assert_log(&[]);
}
