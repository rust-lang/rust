use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use rustc_span::Span;

use crate::query::plumbing::CycleError;
use crate::ty::TyCtxt;

/// A value uniquely identifying an active query job.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct QueryJobId(pub NonZero<u64>);

/// Represents an active query job.
#[derive(Clone, Debug)]
pub struct QueryJob<'tcx> {
    pub id: QueryJobId,

    /// The span corresponding to the reason for which this query was required.
    pub span: Span,

    /// The parent query job which created this job and is implicitly waiting on it.
    pub parent: Option<QueryJobId>,

    /// The latch that is used to wait on this job.
    pub latch: Option<QueryLatch<'tcx>>,
}

impl<'tcx> QueryJob<'tcx> {
    /// Creates a new query job.
    #[inline]
    pub fn new(id: QueryJobId, span: Span, parent: Option<QueryJobId>) -> Self {
        QueryJob { id, span, parent, latch: None }
    }

    pub fn latch(&mut self) -> QueryLatch<'tcx> {
        if self.latch.is_none() {
            self.latch = Some(QueryLatch::new());
        }
        self.latch.as_ref().unwrap().clone()
    }

    /// Signals to waiters that the query is complete.
    ///
    /// This does nothing for single threaded rustc,
    /// as there are no concurrent jobs which could be waiting on us
    #[inline]
    pub fn signal_complete(self) {
        if let Some(latch) = self.latch {
            latch.set();
        }
    }
}

#[derive(Debug)]
pub struct QueryWaiter<'tcx> {
    pub parent: Option<QueryJobId>,
    pub condvar: Condvar,
    pub span: Span,
    pub cycle: Mutex<Option<CycleError<'tcx>>>,
}

#[derive(Clone, Debug)]
pub struct QueryLatch<'tcx> {
    /// The `Option` is `Some(..)` when the job is active, and `None` once completed.
    pub waiters: Arc<Mutex<Option<Vec<Arc<QueryWaiter<'tcx>>>>>>,
}

impl<'tcx> QueryLatch<'tcx> {
    fn new() -> Self {
        QueryLatch { waiters: Arc::new(Mutex::new(Some(Vec::new()))) }
    }

    /// Awaits for the query job to complete.
    pub fn wait_on(
        &self,
        tcx: TyCtxt<'tcx>,
        query: Option<QueryJobId>,
        span: Span,
    ) -> Result<(), CycleError<'tcx>> {
        let mut waiters_guard = self.waiters.lock();
        let Some(waiters) = &mut *waiters_guard else {
            return Ok(()); // already complete
        };

        let waiter = Arc::new(QueryWaiter {
            parent: query,
            span,
            cycle: Mutex::new(None),
            condvar: Condvar::new(),
        });

        // We push the waiter on to the `waiters` list. It can be accessed inside
        // the `wait` call below, by 1) the `set` method or 2) by deadlock detection.
        // Both of these will remove it from the `waiters` list before resuming
        // this thread.
        waiters.push(Arc::clone(&waiter));

        // Awaits the caller on this latch by blocking the current thread.
        // If this detects a deadlock and the deadlock handler wants to resume this thread
        // we have to be in the `wait` call. This is ensured by the deadlock handler
        // getting the self.info lock.
        rustc_thread_pool::mark_blocked();
        tcx.jobserver_proxy.release_thread();
        waiter.condvar.wait(&mut waiters_guard);
        // Release the lock before we potentially block in `acquire_thread`
        drop(waiters_guard);
        tcx.jobserver_proxy.acquire_thread();

        // FIXME: Get rid of this lock. We have ownership of the QueryWaiter
        // although another thread may still have a Arc reference so we cannot
        // use Arc::get_mut
        let mut cycle = waiter.cycle.lock();
        match cycle.take() {
            None => Ok(()),
            Some(cycle) => Err(cycle),
        }
    }

    /// Sets the latch and resumes all waiters on it
    fn set(&self) {
        let mut waiters_guard = self.waiters.lock();
        let waiters = waiters_guard.take().unwrap(); // mark the latch as complete
        let registry = rustc_thread_pool::Registry::current();
        for waiter in waiters {
            rustc_thread_pool::mark_unblocked(&registry);
            waiter.condvar.notify_one();
        }
    }

    /// Removes a single waiter from the list of waiters.
    /// This is used to break query cycles.
    pub fn extract_waiter(&self, waiter: usize) -> Arc<QueryWaiter<'tcx>> {
        let mut waiters_guard = self.waiters.lock();
        let waiters = waiters_guard.as_mut().expect("non-empty waiters vec");
        // Remove the waiter from the list of waiters
        waiters.remove(waiter)
    }
}
