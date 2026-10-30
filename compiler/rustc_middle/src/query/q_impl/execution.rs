use std::hash::Hash;
use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use rustc_data_structures::{defer, outline};
use rustc_errors::FatalError;
use rustc_middle::dep_graph::DepNodeIndex;
use rustc_middle::query::{
    ActiveKeyStatus, QueryCache, QueryCycle, QueryJob, QueryJobId, QueryLatch, QueryState,
    QueryVTable, QueryWaiter,
};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::tls::{self};
use rustc_span::Span;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_structures::Limit;

use super::diagnostics::{QueryOverflow, QueryOverflowNote};
use super::handle_cycle_error;
use crate::query::q_impl::job::{CollectActiveJobsKind, find_cycle_in_stack, find_dep_kind_root};

#[inline]
pub fn equivalent_key<K: Eq, V>(k: K) -> impl Fn(&(K, V)) -> bool {
    move |x| x.0 == k
}

#[cold]
#[inline(never)]
fn handle_cycle<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    cycle: QueryCycle<'tcx>,
) -> C::Value {
    let nested;
    {
        let mut nesting = tcx.query_system.cycle_handler_nesting.lock();
        nested = match *nesting {
            0 => false,
            1 => true,
            _ => {
                // Don't print further nested errors to avoid cases of infinite recursion
                tcx.dcx().delayed_bug("doubly nested cycle error").raise_fatal()
            }
        };
        *nesting += 1;
    }
    let _guard = defer(|| *tcx.query_system.cycle_handler_nesting.lock() -= 1);

    let error = handle_cycle_error::create_cycle_error(tcx, &cycle, nested);

    if nested {
        // Avoid custom handlers and only use the robust `create_cycle_error` for nested cycle errors
        handle_cycle_error::default(error)
    } else {
        (query.handle_cycle_error_fn)(tcx, key, cycle, error)
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
            match shard.find_entry(self.key_hash, equivalent_key(self.key)) {
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

#[cold]
#[inline(never)]
pub fn find_and_handle_cycle<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    try_execute: QueryJobId,
    span: Span,
) -> (C::Value, Option<DepNodeIndex>) {
    // Ensure there were no errors collecting all active jobs.
    // We need the complete map to ensure we find a cycle to break.
    let job_map = tcx.collect_active_query_jobs(CollectActiveJobsKind::FullNoContention);

    let cycle = find_cycle_in_stack(try_execute, job_map, &current_query_job(), span);
    (handle_cycle(query, tcx, key, cycle), None)
}

/// Awaits for the query job to complete.
fn latch_wait_on<'tcx>(
    latch: &QueryLatch<'tcx>,
    query: Option<QueryJobId>,
    span: Span,
) -> Result<(), QueryCycle<'tcx>> {
    let mut waiters_guard = latch.waiters.lock();
    let Some(waiters) = &mut *waiters_guard else {
        return Ok(()); // already complete
    };

    let waiter = Arc::new(QueryWaiter {
        parent: query,
        span,
        cycle: Mutex::new(None),
        condvar: Condvar::new(),
    });

    // We push the waiter on to the `waiters` list. It can be accessed inside the `wait` call
    // below, by 1) the `signal_complete` function or 2) by deadlock detection. Both of these will
    // remove it from the `waiters` list before resuming this thread.
    waiters.push(Arc::clone(&waiter));

    // Awaits the caller on this latch by blocking the current thread. If this detects a deadlock
    // and the deadlock handler wants to resume this thread we have to be in the `wait` call. This
    // is ensured by the deadlock handler getting the waiters lock.
    rustc_thread_pool::mark_blocked_and_wait(|| {
        waiter.condvar.wait(&mut waiters_guard);
        // Release the lock before we potentially block when acquiring jobserver token.
        drop(waiters_guard);
    });

    // FIXME: Get rid of this lock. We have ownership of the QueryWaiter although another thread
    // may still have a Arc reference so we cannot use Arc::get_mut.
    let mut cycle = waiter.cycle.lock();
    match cycle.take() {
        None => Ok(()),
        Some(cycle) => Err(cycle),
    }
}

#[inline(always)]
pub fn wait_for_query<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    key_hash: u64,
    latch: QueryLatch<'tcx>,
    current: Option<QueryJobId>,
) -> (C::Value, Option<DepNodeIndex>) {
    // For parallel queries, we'll block and wait until the query running
    // in another thread has completed. Record how long we wait in the
    // self-profiler.
    let query_blocked_prof_timer = tcx.prof.query_blocked();

    // With parallel queries we might just have to wait on some other thread.
    let result = latch_wait_on(&latch, current, span);

    match result {
        Ok(()) => {
            let Some((v, index)) = query.cache.lookup(&key) else {
                outline(|| {
                    // We didn't find the query result in the query cache. Check if it was
                    // poisoned due to a panic instead.
                    let shard = query.state.active.lock_shard_by_hash(key_hash);
                    match shard.find(key_hash, equivalent_key(key)) {
                        // The query we waited on panicked. Continue unwinding here.
                        Some((_, ActiveKeyStatus::Poisoned)) => FatalError.raise(),
                        _ => panic!(
                            "query '{}' result must be in the cache or the query must be poisoned after a wait",
                            query.name
                        ),
                    }
                })
            };

            tcx.prof.query_cache_hit(index.into());
            query_blocked_prof_timer.finish_with_query_invocation_id(index.into());

            (v, Some(index))
        }
        Err(cycle) => (handle_cycle(query, tcx, key, cycle), None),
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

#[inline(always)]
pub fn check_feedable_consistency<'tcx, C: QueryCache>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
    value: &C::Value,
    existing_value: &C::Value,
) {
    let Some(hash_value_fn) = query.hash_value_fn else {
        panic!(
            "no_hash fed query later has its value computed.\n\
            Remove `no_hash` modifier to allow recomputation.\n\
            The already cached value: {}",
            (query.format_value)(&existing_value)
        );
    };

    let (old_hash, new_hash) = tcx.with_stable_hashing_context(|mut hcx| {
        (hash_value_fn(&mut hcx, &existing_value), hash_value_fn(&mut hcx, value))
    });
    let formatter = query.format_value;
    if old_hash != new_hash {
        // We have an inconsistency. This can happen if one of the two
        // results is tainted by errors.
        assert!(
            tcx.dcx().has_errors().is_some(),
            "Computed query value for {:?}({:?}) is inconsistent with fed value,\n\
                computed={:#?}\nfed={:#?}",
            query.dep_kind,
            key,
            formatter(value),
            formatter(&existing_value),
        );
    }
}

pub fn depth_limit_error<'tcx>(tcx: TyCtxt<'tcx>, job: QueryJobId) {
    let job_map = tcx.collect_active_query_jobs(CollectActiveJobsKind::Full);
    let (span, desc, depth) = find_dep_kind_root(tcx, job, job_map);

    let suggested_limit = match tcx.recursion_limit() {
        Limit(0) => Limit(2),
        limit => limit * 2,
    };

    tcx.dcx().emit_fatal(QueryOverflow {
        span,
        note: QueryOverflowNote { desc, depth },
        suggested_limit,
        crate_name: tcx.crate_name(LOCAL_CRATE),
    });
}
