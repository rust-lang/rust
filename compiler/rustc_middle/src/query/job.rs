use std::fmt::Debug;
use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use rustc_data_structures::hash_table::HashTable;
use rustc_data_structures::sharded::Sharded;
use rustc_span::Span;

use crate::dep_graph::DepNodeIndex;
use crate::queries::TaggedQueryKey;
use crate::query::QueryCache;
use crate::ty::{TyCtxt, tls};

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
}

/// For a particular query and key, tracks the status of a query evaluation
/// that has started, but has not yet finished successfully.
///
/// (Successful query evaluation for a key is represented by an entry in the
/// query's in-memory cache.)
pub enum ActiveKeyStatus<'tcx> {
    /// Some thread is already evaluating the query for this key.
    ///
    /// The enclosed [`QueryJob`] can be used to wait for it to finish.
    Started(QueryJob<'tcx>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error which will
    /// silently panic.
    Poisoned,
}

/// For a particular query, keeps track of "active" keys, i.e. keys whose
/// evaluation has started but has not yet finished successfully.
///
/// (Successful query evaluation for a key is represented by an entry in the
/// query's in-memory cache.)
pub struct QueryState<'tcx, K> {
    pub active: Sharded<HashTable<(K, ActiveKeyStatus<'tcx>)>>,
}

impl<'tcx, K> Default for QueryState<'tcx, K> {
    fn default() -> QueryState<'tcx, K> {
        QueryState { active: Default::default() }
    }
}

/// Description of a frame in the query stack.
///
/// This is mostly used in case of cycles for error reporting.
#[derive(Debug)]
pub struct QueryStackFrame<'tcx> {
    pub span: Span,

    /// The query and key of the query method call that this stack frame
    /// corresponds to.
    ///
    /// Code that doesn't care about the specific key can still use this to
    /// check which query it's for, or obtain the query's name.
    pub tagged_key: TaggedQueryKey<'tcx>,
}

#[derive(Debug)]
pub struct QueryCycle<'tcx> {
    /// The query and related span that uses the cycle.
    pub usage: Option<QueryStackFrame<'tcx>>,

    /// The span here corresponds to the reason for which this query was required.
    pub frames: Vec<QueryStackFrame<'tcx>>,
}

#[derive(Debug)]
pub struct QueryWaiter<'tcx> {
    pub parent: Option<QueryJobId>,
    pub condvar: Condvar,
    pub span: Span,
    pub cycle: Mutex<Option<QueryCycle<'tcx>>>,
}

#[derive(Clone, Debug)]
pub struct QueryLatch<'tcx> {
    /// The `Option` is `Some(..)` when the job is active, and `None` once completed.
    pub waiters: Arc<Mutex<Option<Vec<Arc<QueryWaiter<'tcx>>>>>>,
}

impl<'tcx> QueryLatch<'tcx> {
    pub fn new() -> Self {
        QueryLatch { waiters: Arc::new(Mutex::new(Some(Vec::new()))) }
    }
}

#[inline]
pub fn next_job_id<'tcx>(tcx: TyCtxt<'tcx>) -> QueryJobId {
    QueryJobId(
        NonZero::new(tcx.query_system.jobs.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            .unwrap(),
    )
}

#[inline]
pub fn current_query_job() -> Option<QueryJobId> {
    tls::with_context(|icx| icx.query)
}

/// Guard object representing the responsibility to execute a query job and
/// mark it as completed.
///
/// This will poison the relevant query key if it is dropped without calling
/// [`Self::complete`].
pub struct ActiveJobGuard<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    pub state: &'tcx QueryState<'tcx, K>,
    pub key: K,
    pub key_hash: u64,
}

impl<'tcx, K> ActiveJobGuard<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter, and forgets the guard so it won't poison the query.
    pub fn complete<C>(self, cache: &C, value: C::Value, dep_node_index: DepNodeIndex)
    where
        C: QueryCache<Key = K>,
    {
        // Mark as complete before we remove the job from the active state
        // so no other thread can re-execute this query.
        cache.complete(self.key, value, dep_node_index);

        let mut this = ManuallyDrop::new(self);

        // Drop everything without poisoning the query.
        this.drop_and_maybe_poison(/* poison */ false);
    }

    fn drop_and_maybe_poison(&mut self, poison: bool) {
        let status = {
            let mut shard = self.state.active.lock_shard_by_hash(self.key_hash);
            match shard.find_entry(self.key_hash, |x| x.0 == self.key) {
                Err(_) => {
                    // Note: we must not panic while holding the lock, because unwinding also looks
                    // at this map, which can result in a double panic. So drop it first.
                    drop(shard);
                    panic!();
                }
                Ok(occupied) => {
                    let ((key, status), vacant) = occupied.remove();
                    if poison {
                        vacant.insert((key, ActiveKeyStatus::Poisoned));
                    }
                    status
                }
            }
        };

        // Also signal the completion of the job, so waiters will continue execution.
        match status {
            ActiveKeyStatus::Started(job) => signal_complete(job),
            ActiveKeyStatus::Poisoned => panic!(),
        }
    }
}

impl<'tcx, K> Drop for ActiveJobGuard<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    #[inline(never)]
    #[cold]
    fn drop(&mut self) {
        // Poison the query so jobs waiting on it panic.
        self.drop_and_maybe_poison(/* poison */ true);
    }
}

/// Signals to waiters that the query is complete.
///
/// This does nothing for single threaded rustc, as there are no concurrent jobs which could be
/// waiting on us.
#[inline]
fn signal_complete(job: QueryJob<'_>) {
    if let Some(latch) = job.latch {
        // Set the latch and resume all waiters on it.
        let mut waiters_guard = latch.waiters.lock();
        let waiters = waiters_guard.take().unwrap(); // mark the latch as complete
        let registry = rustc_thread_pool::Registry::current();
        for waiter in waiters {
            rustc_thread_pool::mark_unblocked(&registry);
            waiter.condvar.notify_one();
        }
    }
}
