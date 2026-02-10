use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use rustc_span::Span;

use super::{QueryStackDeferred, QueryStackFrameExtra};
use crate::query::plumbing::CycleError;
use crate::query::{QueryContext, QueryStackFrame};

/// Represents a span and a query key.
#[derive(Clone, Debug)]
pub struct QueryInfo<I> {
    /// The span corresponding to the reason for which this query was required.
    pub span: Span,
    pub frame: QueryStackFrame<I>,
}

impl<'tcx> QueryInfo<QueryStackDeferred<'tcx>> {
    pub(crate) fn lift(&self) -> QueryInfo<QueryStackFrameExtra> {
        QueryInfo { span: self.span, frame: self.frame.lift() }
    }
}

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
    pub query: Option<QueryJobId>,
    pub condvar: Condvar,
    pub span: Span,
    pub cycle: Mutex<Option<CycleError<QueryStackDeferred<'tcx>>>>,
}

#[derive(Debug)]
pub struct QueryLatchInfo<'tcx> {
    pub complete: bool,
    pub waiters: Vec<Arc<QueryWaiter<'tcx>>>,
}

#[derive(Clone, Debug)]
pub struct QueryLatch<'tcx> {
    pub info: Arc<Mutex<QueryLatchInfo<'tcx>>>,
}

impl<'tcx> QueryLatch<'tcx> {
    fn new() -> Self {
        QueryLatch {
            info: Arc::new(Mutex::new(QueryLatchInfo { complete: false, waiters: Vec::new() })),
        }
    }

    /// Awaits for the query job to complete.
    pub fn wait_on(
        &self,
        qcx: impl QueryContext<'tcx>,
        query: Option<QueryJobId>,
        span: Span,
    ) -> Result<(), CycleError<QueryStackDeferred<'tcx>>> {
        let waiter =
            Arc::new(QueryWaiter { query, span, cycle: Mutex::new(None), condvar: Condvar::new() });
        self.wait_on_inner(qcx, &waiter);
        // FIXME: Get rid of this lock. We have ownership of the QueryWaiter
        // although another thread may still have a Arc reference so we cannot
        // use Arc::get_mut
        let mut cycle = waiter.cycle.lock();
        match cycle.take() {
            None => Ok(()),
            Some(cycle) => Err(cycle),
        }
    }

    /// Awaits the caller on this latch by blocking the current thread.
    fn wait_on_inner(&self, qcx: impl QueryContext<'tcx>, waiter: &Arc<QueryWaiter<'tcx>>) {
        let mut info = self.info.lock();
        if !info.complete {
            // We push the waiter on to the `waiters` list. It can be accessed inside
            // the `wait` call below, by 1) the `set` method or 2) by deadlock detection.
            // Both of these will remove it from the `waiters` list before resuming
            // this thread.
            info.waiters.push(Arc::clone(waiter));

            // If this detects a deadlock and the deadlock handler wants to resume this thread
            // we have to be in the `wait` call. This is ensured by the deadlock handler
            // getting the self.info lock.
            rustc_thread_pool::mark_blocked();
            let proxy = qcx.jobserver_proxy();
            proxy.release_thread();
            waiter.condvar.wait(&mut info);
            // Release the lock before we potentially block in `acquire_thread`
            drop(info);
            proxy.acquire_thread();
        }
    }

    /// Sets the latch and resumes all waiters on it
    fn set(&self) {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        info.complete = true;
        let registry = rustc_thread_pool::Registry::current();
        for waiter in info.waiters.drain(..) {
            rustc_thread_pool::mark_unblocked(&registry);
            waiter.condvar.notify_one();
        }
    }

    /// Removes a single waiter from the list of waiters.
    /// This is used to break query cycles.
    pub fn extract_waiter(&self, waiter: usize) -> Arc<QueryWaiter<'tcx>> {
        let mut info = self.info.lock();
        debug_assert!(!info.complete);
        // Remove the waiter from the list of waiters
        info.waiters.remove(waiter)
    }
}
