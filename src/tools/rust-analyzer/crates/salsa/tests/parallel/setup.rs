use crate::signal::Signal;
use salsa::Database;
use salsa::ParallelDatabase;
use salsa::Snapshot;
use std::sync::Arc;
use std::{
    cell::Cell,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
};

#[salsa::query_group(Par)]
pub(crate) trait ParDatabase: Knobs {
    #[salsa::input]
    fn input(&self, key: char) -> usize;

    fn sum(&self, key: &'static str) -> usize;

    /// Invokes `sum`
    fn sum2(&self, key: &'static str) -> usize;

    /// Invokes `sum` but doesn't really care about the result.
    fn sum2_drop_sum(&self, key: &'static str) -> usize;

    /// Invokes `sum2`
    fn sum3(&self, key: &'static str) -> usize;

    /// Invokes `sum2_drop_sum`
    fn sum3_drop_sum(&self, key: &'static str) -> usize;
}

/// Various "knobs" and utilities used by tests to force
/// a certain behavior.
pub(crate) trait Knobs {
    fn knobs(&self) -> &KnobsStruct;

    fn signal(&self, stage: usize);

    fn wait_for(&self, stage: usize);
}

pub(crate) trait WithValue<T> {
    fn with_value<R>(&self, value: T, closure: impl FnOnce() -> R) -> R;
}

impl<T> WithValue<T> for Cell<T> {
    fn with_value<R>(&self, value: T, closure: impl FnOnce() -> R) -> R {
        let old_value = self.replace(value);

        let result = catch_unwind(AssertUnwindSafe(closure));

        self.set(old_value);

        match result {
            Ok(r) => r,
            Err(payload) => resume_unwind(payload),
        }
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancellationFlag {
    #[default]
    Down,
    Panic,
}

/// Various "knobs" that can be used to customize how the queries
/// behave on one specific thread. Note that this state is
/// intentionally thread-local (apart from `signal`).
#[derive(Clone, Default)]
pub(crate) struct KnobsStruct {
    /// A kind of flexible barrier used to coordinate execution across
    /// threads to ensure we reach various weird states.
    pub(crate) signal: Arc<Signal>,

    /// When this database is about to block, send a signal.
    pub(crate) signal_on_will_block: Cell<usize>,

    /// Invocations of `sum` will signal this stage on entry.
    pub(crate) sum_signal_on_entry: Cell<usize>,

    /// Invocations of `sum` will wait for this stage on entry.
    pub(crate) sum_wait_for_on_entry: Cell<usize>,

    /// If true, invocations of `sum` will panic before they exit.
    pub(crate) sum_should_panic: Cell<bool>,

    /// If true, invocations of `sum` will wait for cancellation before
    /// they exit.
    pub(crate) sum_wait_for_cancellation: Cell<CancellationFlag>,

    /// Invocations of `sum` will wait for this stage prior to exiting.
    pub(crate) sum_wait_for_on_exit: Cell<usize>,

    /// Invocations of `sum` will signal this stage prior to exiting.
    pub(crate) sum_signal_on_exit: Cell<usize>,

    /// Invocations of `sum3_drop_sum` will panic unconditionally
    pub(crate) sum3_drop_sum_should_panic: Cell<bool>,
}

fn sum(db: &dyn ParDatabase, key: &'static str) -> usize {
    let mut sum = 0;

    db.signal(db.knobs().sum_signal_on_entry.get());

    db.wait_for(db.knobs().sum_wait_for_on_entry.get());

    if db.knobs().sum_should_panic.get() {
        panic!("query set to panic before exit")
    }

    for ch in key.chars() {
        sum += db.input(ch);
    }

    match db.knobs().sum_wait_for_cancellation.get() {
        CancellationFlag::Down => (),
        CancellationFlag::Panic => {
            tracing::debug!("waiting for cancellation");
            loop {
                db.unwind_if_cancelled();
                std::thread::yield_now();
            }
        }
    }

    db.wait_for(db.knobs().sum_wait_for_on_exit.get());

    db.signal(db.knobs().sum_signal_on_exit.get());

    sum
}

fn sum2(db: &dyn ParDatabase, key: &'static str) -> usize {
    db.sum(key)
}

fn sum2_drop_sum(db: &dyn ParDatabase, key: &'static str) -> usize {
    let _ = db.sum(key);
    22
}

fn sum3(db: &dyn ParDatabase, key: &'static str) -> usize {
    db.sum2(key)
}

fn sum3_drop_sum(db: &dyn ParDatabase, key: &'static str) -> usize {
    if db.knobs().sum3_drop_sum_should_panic.get() {
        panic!("sum3_drop_sum executed")
    }
    db.sum2_drop_sum(key)
}

#[salsa::database(
    Par,
    crate::parallel_cycle_all_recover::ParallelCycleAllRecover,
    crate::parallel_cycle_none_recover::ParallelCycleNoneRecover,
    crate::parallel_cycle_mid_recover::ParallelCycleMidRecovers,
    crate::parallel_cycle_one_recovers::ParallelCycleOneRecovers
)]
#[derive(Default)]
pub(crate) struct ParDatabaseImpl {
    storage: salsa::Storage<Self>,
    knobs: KnobsStruct,
}

impl Database for ParDatabaseImpl {
    fn salsa_event(&self, event: salsa::Event) {
        if let salsa::EventKind::WillBlockOn { .. } = event.kind {
            self.signal(self.knobs().signal_on_will_block.get());
        }
    }
}

impl ParallelDatabase for ParDatabaseImpl {
    fn snapshot(&self) -> Snapshot<Self> {
        Snapshot::new(ParDatabaseImpl {
            storage: self.storage.snapshot(),
            knobs: self.knobs.clone(),
        })
    }
}

impl Knobs for ParDatabaseImpl {
    fn knobs(&self) -> &KnobsStruct {
        &self.knobs
    }

    fn signal(&self, stage: usize) {
        self.knobs.signal.signal(stage);
    }

    fn wait_for(&self, stage: usize) {
        self.knobs.signal.wait_for(stage);
    }
}
