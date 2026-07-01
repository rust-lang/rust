use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::Mutex;
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
pub struct QueryWaiter<'tcx> {
    pub parent: Option<QueryJobId>,
    pub span: Span,
    pub cycle: Option<Cycle<'tcx>>,
}

#[derive(Clone, Debug)]
pub struct QueryLatch<'tcx> {
    /// The `waiters` is not `usize::MAX` when the job is active, and `usize::MAX` once completed.
    pub waiters: Arc<Mutex<usize>>,
    pub _marker: PhantomData<&'tcx ()>,
}

impl<'tcx> QueryLatch<'tcx> {
    fn new() -> Self {
        QueryLatch { waiters: Arc::new(Mutex::new(0)), _marker: PhantomData }
    }

    /// Awaits for the query job to complete.
    pub fn wait_on(
        &self,
        tcx: TyCtxt<'tcx>,
        query: Option<QueryJobId>,
        span: Span,
    ) -> Result<(), Cycle<'tcx>> {
        let thread_index = rustc_thread_pool::current_thread_index().unwrap();
        let mut waiters_guard = self.waiters.lock();
        if *waiters_guard == usize::MAX {
            return Ok(()); // already complete
        };
        debug_assert!(*waiters_guard & (1 << thread_index) == 0);

        let waiter = QueryWaiter { parent: query, span, cycle: None };

        // We push the waiter on to the `waiters` list. It can be accessed inside
        // the `wait` call below, by 1) the `set` method or 2) by deadlock detection.
        // Both of these will remove it from the `waiters` list before resuming
        // this thread.
        let mut waiters_state = tcx.waiters.lock();
        if mem::replace(&mut *waiters_state, Some(waiter)).is_some() {
            panic!("tried to place a waiter twice for a worker thread")
        }
        *waiters_guard |= 1 << thread_index;
        drop(waiters_state);

        // Awaits the caller on this latch by blocking the current thread.
        // If this detects a deadlock and the deadlock handler wants to resume this thread
        // we have to be in the `wait` call. This is ensured by the deadlock handler
        // getting the self.info lock.
        rustc_thread_pool::park(waiters_guard, |_| {
            // Reset our QueryWaiter to None
            let waiter = tcx.waiters.lock().take().unwrap();
            match waiter.cycle {
                None => Ok(()),
                Some(cycle) => Err(cycle),
            }
        })
    }

    /// Sets the latch and resumes all waiters on it
    fn set(&self) {
        let mut waiters_guard = self.waiters.lock();
        let waiters = mem::replace(&mut *waiters_guard, usize::MAX); // mark the latch as complete
        debug_assert!(waiters != usize::MAX);
        let registry = rustc_thread_pool::Registry::current();
        for waiter_thread in 0..usize::BITS - 1 {
            if waiters & (1 << waiter_thread) != 0 {
                rustc_thread_pool::unpark(&registry, waiter_thread as usize);
            }
        }
    }
}
