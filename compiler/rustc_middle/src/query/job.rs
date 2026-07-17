use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex, MutexGuard};
use rustc_span::Span;

use crate::query::Cycle;
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
        self.latch.get_or_insert_with(QueryLatch::new).clone()
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
pub struct QueryWaiter {
    pub parent: Option<QueryJobId>,
    pub condvar: Arc<Condvar>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct QueryLatch<'tcx> {
    /// The `Option` is `Some(..)` when the job is active, and `None` once completed.
    pub inner: Arc<Mutex<Option<QueryLatchState<'tcx>>>>,
}

#[derive(Debug)]
pub struct QueryLatchState<'tcx> {
    pub waiters: Vec<QueryWaiter>,
    pub cycle: Option<(usize, Cycle<'tcx>)>,
}

impl<'tcx> QueryLatch<'tcx> {
    fn new() -> Self {
        QueryLatch {
            inner: Arc::new(Mutex::new(Some(QueryLatchState { waiters: Vec::new(), cycle: None }))),
        }
    }

    /// Awaits for the query job to complete.
    pub fn wait_on(
        &self,
        tcx: TyCtxt<'tcx>,
        parent: Option<QueryJobId>,
        span: Span,
    ) -> Result<(), Cycle<'tcx>> {
        let mut state_lock = self.inner.lock();
        let Some(state) = &mut *state_lock else {
            return Ok(()); // already complete
        };

        let condvar = Arc::new(Condvar::new());
        let waiter = QueryWaiter { parent, span, condvar: Arc::clone(&condvar) };

        state.waiters.reserve(state.waiters.len().saturating_sub(tcx.sess.threads()));
        // We push the waiter on to the `waiters` list. It can be accessed inside
        // the `wait` call below, by 1) the `set` method or 2) by deadlock detection.
        // Both of these will remove it from the `waiters` list before resuming
        // this thread.
        state.waiters.push(waiter);

        // Awaits the caller on this latch by blocking the current thread.
        // If this detects a deadlock and the deadlock handler wants to resume this thread
        // we have to be in the `wait` call. This is ensured by the deadlock handler
        // getting the self.info lock.
        let cycle = rustc_thread_pool::Registry::with_current(|registry| {
            let handler = rustc_thread_pool::mark_blocked(registry);
            tcx.jobserver_proxy.release_thread();
            if let Some(handler) = handler {
                MutexGuard::unlocked(&mut state_lock, handler);
            }

            let cv_addr = (&*condvar as *const Condvar).addr();
            if handler.is_some()
                && let Some((_, cycle)) = state_lock
                    .as_mut()?
                    .cycle
                    .take_if(|(resumed_cv_addr, _)| *resumed_cv_addr == cv_addr)
            {
                return Some(cycle);
            }

            condvar.wait(&mut state_lock);
            let (resumed_cv_addr, cycle) = state_lock
                .as_mut()?
                .cycle
                .take()
                .expect("resumed waiter for unfinished query without a cycle");
            assert_eq!(resumed_cv_addr, cv_addr);
            Some(cycle)
        });

        // Release the lock before we potentially block in `acquire_thread`
        drop(state_lock);
        tcx.jobserver_proxy.acquire_thread();

        match cycle {
            None => Ok(()),
            Some(cycle) => Err(cycle),
        }
    }

    /// Sets the latch and resumes all waiters on it
    fn set(&self) {
        let mut state_lock = self.inner.lock();
        let waiters = state_lock.take().unwrap().waiters; // mark the latch as complete
        rustc_thread_pool::Registry::with_current(|registry| {
            for waiter in waiters {
                rustc_thread_pool::mark_unblocked(&registry);
                waiter.condvar.notify_one();
            }
        });
    }
}
