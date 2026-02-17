use std::hash::Hash;
use std::mem;

use rustc_data_structures::hash_table::{Entry, HashTable};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::{outline, sharded, sync};
use rustc_errors::{Diag, FatalError, StashKey};
use rustc_middle::dep_graph::{DepGraphData, DepNodeKey, HasDepContext};
use rustc_middle::query::{
    ActiveKeyStatus, CycleError, CycleErrorHandling, QueryCache, QueryJob, QueryJobId, QueryLatch,
    QueryMode, QueryStackDeferred, QueryStackFrame, QueryState,
};
use rustc_middle::ty::TyCtxt;
use rustc_middle::verify_ich::incremental_verify_ich;
use rustc_span::{DUMMY_SP, Span};

use crate::dep_graph::{DepNode, DepNodeIndex};
use crate::job::{QueryJobInfo, QueryJobMap, find_cycle_in_stack, report_cycle};
use crate::{QueryCtxt, QueryFlags, SemiDynamicQueryDispatcher};

#[inline]
fn equivalent_key<K: Eq, V>(k: &K) -> impl Fn(&(K, V)) -> bool + '_ {
    move |x| x.0 == *k
}

/// Obtains the enclosed [`QueryJob`], or panics if this query evaluation
/// was poisoned by a panic.
fn expect_job<'tcx>(status: ActiveKeyStatus<'tcx>) -> QueryJob<'tcx> {
    match status {
        ActiveKeyStatus::Started(job) => job,
        ActiveKeyStatus::Poisoned => {
            panic!("job for query failed to start and was poisoned")
        }
    }
}

pub(crate) fn all_inactive<'tcx, K>(state: &QueryState<'tcx, K>) -> bool {
    state.active.lock_shards().all(|shard| shard.is_empty())
}

/// Internal plumbing for collecting the set of active jobs for this query.
///
/// Should only be called from `gather_active_jobs`.
pub(crate) fn gather_active_jobs_inner<'tcx, K: Copy>(
    state: &QueryState<'tcx, K>,
    tcx: TyCtxt<'tcx>,
    make_frame: fn(TyCtxt<'tcx>, K) -> QueryStackFrame<QueryStackDeferred<'tcx>>,
    require_complete: bool,
    job_map_out: &mut QueryJobMap<'tcx>, // Out-param; job info is gathered into this map
) -> Option<()> {
    let mut active = Vec::new();

    // Helper to gather active jobs from a single shard.
    let mut gather_shard_jobs = |shard: &HashTable<(K, ActiveKeyStatus<'tcx>)>| {
        for (k, v) in shard.iter() {
            if let ActiveKeyStatus::Started(ref job) = *v {
                active.push((*k, job.clone()));
            }
        }
    };

    // Lock shards and gather jobs from each shard.
    if require_complete {
        for shard in state.active.lock_shards() {
            gather_shard_jobs(&shard);
        }
    } else {
        // We use try_lock_shards here since we are called from the
        // deadlock handler, and this shouldn't be locked.
        for shard in state.active.try_lock_shards() {
            let shard = shard?;
            gather_shard_jobs(&shard);
        }
    }

    // Call `make_frame` while we're not holding a `state.active` lock as `make_frame` may call
    // queries leading to a deadlock.
    for (key, job) in active {
        let frame = make_frame(tcx, key);
        job_map_out.insert(job.id, QueryJobInfo { frame, job });
    }

    Some(())
}

/// Guard object representing the responsibility to execute a query job and
/// mark it as completed.
///
/// This will poison the relevant query key if it is dropped without calling
/// [`Self::complete`].
struct ActiveJobGuard<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    state: &'tcx QueryState<'tcx, K>,
    key: K,
    key_hash: u64,
}

#[cold]
#[inline(never)]
fn mk_cycle<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    cycle_error: CycleError,
) -> C::Value {
    let error = report_cycle(qcx.tcx.sess, &cycle_error);
    handle_cycle_error(query, qcx, &cycle_error, error)
}

fn handle_cycle_error<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    cycle_error: &CycleError,
    error: Diag<'_>,
) -> C::Value {
    match query.cycle_error_handling() {
        CycleErrorHandling::Error => {
            let guar = error.emit();
            query.value_from_cycle_error(qcx.tcx, cycle_error, guar)
        }
        CycleErrorHandling::Fatal => {
            error.emit();
            qcx.tcx.dcx().abort_if_errors();
            unreachable!()
        }
        CycleErrorHandling::DelayBug => {
            let guar = error.delay_as_bug();
            query.value_from_cycle_error(qcx.tcx, cycle_error, guar)
        }
        CycleErrorHandling::Stash => {
            let guar = if let Some(root) = cycle_error.cycle.first()
                && let Some(span) = root.frame.info.span
            {
                error.stash(span, StashKey::Cycle).unwrap()
            } else {
                error.emit()
            };
            query.value_from_cycle_error(qcx.tcx, cycle_error, guar)
        }
    }
}

impl<'tcx, K> ActiveJobGuard<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter, and forgets the guard so it won't poison the query.
    fn complete<C>(self, cache: &C, result: C::Value, dep_node_index: DepNodeIndex)
    where
        C: QueryCache<Key = K>,
    {
        // Forget ourself so our destructor won't poison the query.
        // (Extract fields by value first to make sure we don't leak anything.)
        let Self { state, key, key_hash }: Self = self;
        mem::forget(self);

        // Mark as complete before we remove the job from the active state
        // so no other thread can re-execute this query.
        cache.complete(key, result, dep_node_index);

        let job = {
            // don't keep the lock during the `unwrap()` of the retrieved value, or we taint the
            // underlying shard.
            // since unwinding also wants to look at this map, this can also prevent a double
            // panic.
            let mut shard = state.active.lock_shard_by_hash(key_hash);
            match shard.find_entry(key_hash, equivalent_key(&key)) {
                Err(_) => None,
                Ok(occupied) => Some(occupied.remove().0.1),
            }
        };
        let job = expect_job(job.expect("active query job entry"));

        job.signal_complete();
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
        let Self { state, key, key_hash } = *self;
        let job = {
            let mut shard = state.active.lock_shard_by_hash(key_hash);
            match shard.find_entry(key_hash, equivalent_key(&key)) {
                Err(_) => panic!(),
                Ok(occupied) => {
                    let ((key, value), vacant) = occupied.remove();
                    vacant.insert((key, ActiveKeyStatus::Poisoned));
                    expect_job(value)
                }
            }
        };
        // Also signal the completion of the job, so waiters
        // will continue execution.
        job.signal_complete();
    }
}

#[cold]
#[inline(never)]
fn cycle_error<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    try_execute: QueryJobId,
    span: Span,
) -> (C::Value, Option<DepNodeIndex>) {
    // Ensure there was no errors collecting all active jobs.
    // We need the complete map to ensure we find a cycle to break.
    let job_map = qcx
        .collect_active_jobs_from_all_queries(false)
        .ok()
        .expect("failed to collect active queries");

    let error = find_cycle_in_stack(try_execute, job_map, &qcx.current_query_job(), span);
    (mk_cycle(query, qcx, error.lift()), None)
}

#[inline(always)]
fn wait_for_query<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    span: Span,
    key: C::Key,
    latch: QueryLatch<'tcx>,
    current: Option<QueryJobId>,
) -> (C::Value, Option<DepNodeIndex>) {
    // For parallel queries, we'll block and wait until the query running
    // in another thread has completed. Record how long we wait in the
    // self-profiler.
    let query_blocked_prof_timer = qcx.tcx.prof.query_blocked();

    // With parallel queries we might just have to wait on some other
    // thread.
    let result = latch.wait_on(qcx.tcx, current, span);

    match result {
        Ok(()) => {
            let Some((v, index)) = query.query_cache(qcx).lookup(&key) else {
                outline(|| {
                    // We didn't find the query result in the query cache. Check if it was
                    // poisoned due to a panic instead.
                    let key_hash = sharded::make_hash(&key);
                    let shard = query.query_state(qcx).active.lock_shard_by_hash(key_hash);
                    match shard.find(key_hash, equivalent_key(&key)) {
                        // The query we waited on panicked. Continue unwinding here.
                        Some((_, ActiveKeyStatus::Poisoned)) => FatalError.raise(),
                        _ => panic!(
                            "query '{}' result must be in the cache or the query must be poisoned after a wait",
                            query.name()
                        ),
                    }
                })
            };

            qcx.tcx.prof.query_cache_hit(index.into());
            query_blocked_prof_timer.finish_with_query_invocation_id(index.into());

            (v, Some(index))
        }
        Err(cycle) => (mk_cycle(query, qcx, cycle.lift()), None),
    }
}

#[inline(never)]
fn try_execute_query<'tcx, C: QueryCache, const FLAGS: QueryFlags, const INCR: bool>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    span: Span,
    key: C::Key,
    dep_node: Option<DepNode>,
) -> (C::Value, Option<DepNodeIndex>) {
    let state = query.query_state(qcx);
    let key_hash = sharded::make_hash(&key);
    let mut state_lock = state.active.lock_shard_by_hash(key_hash);

    // For the parallel compiler we need to check both the query cache and query state structures
    // while holding the state lock to ensure that 1) the query has not yet completed and 2) the
    // query is not still executing. Without checking the query cache here, we can end up
    // re-executing the query since `try_start` only checks that the query is not currently
    // executing, but another thread may have already completed the query and stores it result
    // in the query cache.
    if qcx.tcx.sess.threads() > 1 {
        if let Some((value, index)) = query.query_cache(qcx).lookup(&key) {
            qcx.tcx.prof.query_cache_hit(index.into());
            return (value, Some(index));
        }
    }

    let current_job_id = qcx.current_query_job();

    match state_lock.entry(key_hash, equivalent_key(&key), |(k, _)| sharded::make_hash(k)) {
        Entry::Vacant(entry) => {
            // Nothing has computed or is computing the query, so we start a new job and insert it in the
            // state map.
            let id = qcx.next_job_id();
            let job = QueryJob::new(id, span, current_job_id);
            entry.insert((key, ActiveKeyStatus::Started(job)));

            // Drop the lock before we start executing the query
            drop(state_lock);

            execute_job::<C, FLAGS, INCR>(query, qcx, state, key, key_hash, id, dep_node)
        }
        Entry::Occupied(mut entry) => {
            match &mut entry.get_mut().1 {
                ActiveKeyStatus::Started(job) => {
                    if sync::is_dyn_thread_safe() {
                        // Get the latch out
                        let latch = job.latch();
                        drop(state_lock);

                        // Only call `wait_for_query` if we're using a Rayon thread pool
                        // as it will attempt to mark the worker thread as blocked.
                        return wait_for_query(query, qcx, span, key, latch, current_job_id);
                    }

                    let id = job.id;
                    drop(state_lock);

                    // If we are single-threaded we know that we have cycle error,
                    // so we just return the error.
                    cycle_error(query, qcx, id, span)
                }
                ActiveKeyStatus::Poisoned => FatalError.raise(),
            }
        }
    }
}

#[inline(always)]
fn execute_job<'tcx, C: QueryCache, const FLAGS: QueryFlags, const INCR: bool>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    state: &'tcx QueryState<'tcx, C::Key>,
    key: C::Key,
    key_hash: u64,
    id: QueryJobId,
    dep_node: Option<DepNode>,
) -> (C::Value, Option<DepNodeIndex>) {
    // Set up a guard object that will automatically poison the query if a
    // panic occurs while executing the query (or any intermediate plumbing).
    let job_guard = ActiveJobGuard { state, key, key_hash };

    debug_assert_eq!(qcx.tcx.dep_graph.is_fully_enabled(), INCR);

    // Delegate to another function to actually execute the query job.
    let (result, dep_node_index) = if INCR {
        execute_job_incr(query, qcx, qcx.tcx.dep_graph.data().unwrap(), key, dep_node, id)
    } else {
        execute_job_non_incr(query, qcx, key, id)
    };

    let cache = query.query_cache(qcx);
    if query.feedable() {
        // We should not compute queries that also got a value via feeding.
        // This can't happen, as query feeding adds the very dependencies to the fed query
        // as its feeding query had. So if the fed query is red, so is its feeder, which will
        // get evaluated first, and re-feed the query.
        if let Some((cached_result, _)) = cache.lookup(&key) {
            let Some(hasher) = query.hash_result() else {
                panic!(
                    "no_hash fed query later has its value computed.\n\
                    Remove `no_hash` modifier to allow recomputation.\n\
                    The already cached value: {}",
                    (query.format_value())(&cached_result)
                );
            };

            let (old_hash, new_hash) = qcx.dep_context().with_stable_hashing_context(|mut hcx| {
                (hasher(&mut hcx, &cached_result), hasher(&mut hcx, &result))
            });
            let formatter = query.format_value();
            if old_hash != new_hash {
                // We have an inconsistency. This can happen if one of the two
                // results is tainted by errors.
                assert!(
                    qcx.tcx.dcx().has_errors().is_some(),
                    "Computed query value for {:?}({:?}) is inconsistent with fed value,\n\
                        computed={:#?}\nfed={:#?}",
                    query.dep_kind(),
                    key,
                    formatter(&result),
                    formatter(&cached_result),
                );
            }
        }
    }

    // Tell the guard to perform completion bookkeeping, and also to not poison the query.
    job_guard.complete(cache, result, dep_node_index);

    (result, Some(dep_node_index))
}

// Fast path for when incr. comp. is off.
#[inline(always)]
fn execute_job_non_incr<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    key: C::Key,
    job_id: QueryJobId,
) -> (C::Value, DepNodeIndex) {
    debug_assert!(!qcx.tcx.dep_graph.is_fully_enabled());

    // Fingerprint the key, just to assert that it doesn't
    // have anything we don't consider hashable
    if cfg!(debug_assertions) {
        let _ = key.to_fingerprint(qcx.tcx);
    }

    let prof_timer = qcx.tcx.prof.query_provider();
    // Call the query provider.
    let result = qcx.start_query(job_id, query.depth_limit(), || query.invoke_provider(qcx, key));
    let dep_node_index = qcx.tcx.dep_graph.next_virtual_depnode_index();
    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    // Similarly, fingerprint the result to assert that
    // it doesn't have anything not considered hashable.
    if cfg!(debug_assertions)
        && let Some(hash_result) = query.hash_result()
    {
        qcx.dep_context().with_stable_hashing_context(|mut hcx| {
            hash_result(&mut hcx, &result);
        });
    }

    (result, dep_node_index)
}

#[inline(always)]
fn execute_job_incr<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    dep_graph_data: &DepGraphData,
    key: C::Key,
    mut dep_node_opt: Option<DepNode>,
    job_id: QueryJobId,
) -> (C::Value, DepNodeIndex) {
    if !query.anon() && !query.eval_always() {
        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node = dep_node_opt.get_or_insert_with(|| query.construct_dep_node(qcx.tcx, &key));

        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = qcx.start_query(job_id, false, || {
            try_load_from_disk_and_cache_in_memory(query, dep_graph_data, qcx, &key, dep_node)
        }) {
            return ret;
        }
    }

    let prof_timer = qcx.tcx.prof.query_provider();

    let (result, dep_node_index) = qcx.start_query(job_id, query.depth_limit(), || {
        if query.anon() {
            // Call the query provider inside an anon task.
            return dep_graph_data.with_anon_task_inner(qcx.tcx, query.dep_kind(), || {
                query.invoke_provider(qcx, key)
            });
        }

        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node = dep_node_opt.unwrap_or_else(|| query.construct_dep_node(qcx.tcx, &key));

        // Call the query provider.
        dep_graph_data.with_task(
            dep_node,
            (qcx, query),
            key,
            |(qcx, query), key| query.invoke_provider(qcx, key),
            query.hash_result(),
        )
    });

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    (result, dep_node_index)
}

#[inline(always)]
fn try_load_from_disk_and_cache_in_memory<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    dep_graph_data: &DepGraphData,
    qcx: QueryCtxt<'tcx>,
    key: &C::Key,
    dep_node: &DepNode,
) -> Option<(C::Value, DepNodeIndex)> {
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    let (prev_dep_node_index, dep_node_index) = dep_graph_data.try_mark_green(qcx.tcx, dep_node)?;

    debug_assert!(dep_graph_data.is_index_green(prev_dep_node_index));

    // First we try to load the result from the on-disk cache.
    // Some things are never cached on disk.
    if let Some(result) = query.try_load_from_disk(qcx, key, prev_dep_node_index, dep_node_index) {
        if std::intrinsics::unlikely(qcx.tcx.sess.opts.unstable_opts.query_dep_graph) {
            dep_graph_data.mark_debug_loaded_from_disk(*dep_node)
        }

        let prev_fingerprint = dep_graph_data.prev_fingerprint_of(prev_dep_node_index);
        // If `-Zincremental-verify-ich` is specified, re-hash results from
        // the cache and make sure that they have the expected fingerprint.
        //
        // If not, we still seek to verify a subset of fingerprints loaded
        // from disk. Re-hashing results is fairly expensive, so we can't
        // currently afford to verify every hash. This subset should still
        // give us some coverage of potential bugs though.
        let try_verify = prev_fingerprint.split().1.as_u64().is_multiple_of(32);
        if std::intrinsics::unlikely(
            try_verify || qcx.tcx.sess.opts.unstable_opts.incremental_verify_ich,
        ) {
            incremental_verify_ich(
                qcx.tcx,
                dep_graph_data,
                &result,
                prev_dep_node_index,
                query.hash_result(),
                query.format_value(),
            );
        }

        return Some((result, dep_node_index));
    }

    // We always expect to find a cached result for things that
    // can be forced from `DepNode`.
    debug_assert!(
        !query.will_cache_on_disk_for_key(qcx.tcx, key)
            || !qcx.tcx.fingerprint_style(dep_node.kind).reconstructible(),
        "missing on-disk cache entry for {dep_node:?}"
    );

    // Sanity check for the logic in `ensure`: if the node is green and the result loadable,
    // we should actually be able to load it.
    debug_assert!(
        !query.is_loadable_from_disk(qcx, key, prev_dep_node_index),
        "missing on-disk cache entry for loadable {dep_node:?}"
    );

    // We could not load a result from the on-disk cache, so
    // recompute.
    let prof_timer = qcx.tcx.prof.query_provider();

    // The dep-graph for this computation is already in-place.
    // Call the query provider.
    let result = qcx.tcx.dep_graph.with_ignore(|| query.invoke_provider(qcx, *key));

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    // Verify that re-running the query produced a result with the expected hash
    // This catches bugs in query implementations, turning them into ICEs.
    // For example, a query might sort its result by `DefId` - since `DefId`s are
    // not stable across compilation sessions, the result could get up getting sorted
    // in a different order when the query is re-run, even though all of the inputs
    // (e.g. `DefPathHash` values) were green.
    //
    // See issue #82920 for an example of a miscompilation that would get turned into
    // an ICE by this check
    incremental_verify_ich(
        qcx.tcx,
        dep_graph_data,
        &result,
        prev_dep_node_index,
        query.hash_result(),
        query.format_value(),
    );

    Some((result, dep_node_index))
}

/// Ensure that either this query has all green inputs or been executed.
/// Executing `query::ensure(D)` is considered a read of the dep-node `D`.
/// Returns true if the query should still run.
///
/// This function is particularly useful when executing passes for their
/// side-effects -- e.g., in order to report errors for erroneous programs.
///
/// Note: The optimization is only available during incr. comp.
#[inline(never)]
fn ensure_must_run<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    key: &C::Key,
    check_cache: bool,
) -> (bool, Option<DepNode>) {
    if query.eval_always() {
        return (true, None);
    }

    // Ensuring an anonymous query makes no sense
    assert!(!query.anon());

    let dep_node = query.construct_dep_node(qcx.tcx, key);

    let dep_graph = &qcx.tcx.dep_graph;
    let serialized_dep_node_index = match dep_graph.try_mark_green(qcx.tcx, &dep_node) {
        None => {
            // A None return from `try_mark_green` means that this is either
            // a new dep node or that the dep node has already been marked red.
            // Either way, we can't call `dep_graph.read()` as we don't have the
            // DepNodeIndex. We must invoke the query itself. The performance cost
            // this introduces should be negligible as we'll immediately hit the
            // in-memory cache, or another query down the line will.
            return (true, Some(dep_node));
        }
        Some((serialized_dep_node_index, dep_node_index)) => {
            dep_graph.read_index(dep_node_index);
            qcx.tcx.prof.query_cache_hit(dep_node_index.into());
            serialized_dep_node_index
        }
    };

    // We do not need the value at all, so do not check the cache.
    if !check_cache {
        return (false, None);
    }

    let loadable = query.is_loadable_from_disk(qcx, key, serialized_dep_node_index);
    (!loadable, Some(dep_node))
}

#[inline(always)]
pub(super) fn get_query_non_incr<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    span: Span,
    key: C::Key,
) -> C::Value {
    debug_assert!(!qcx.tcx.dep_graph.is_fully_enabled());

    ensure_sufficient_stack(|| try_execute_query::<C, FLAGS, false>(query, qcx, span, key, None).0)
}

#[inline(always)]
pub(super) fn get_query_incr<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    span: Span,
    key: C::Key,
    mode: QueryMode,
) -> Option<C::Value> {
    debug_assert!(qcx.tcx.dep_graph.is_fully_enabled());

    let dep_node = if let QueryMode::Ensure { check_cache } = mode {
        let (must_run, dep_node) = ensure_must_run(query, qcx, &key, check_cache);
        if !must_run {
            return None;
        }
        dep_node
    } else {
        None
    };

    let (result, dep_node_index) = ensure_sufficient_stack(|| {
        try_execute_query::<C, FLAGS, true>(query, qcx, span, key, dep_node)
    });
    if let Some(dep_node_index) = dep_node_index {
        qcx.tcx.dep_graph.read_index(dep_node_index)
    }
    Some(result)
}

pub(crate) fn force_query<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    key: C::Key,
    dep_node: DepNode,
) {
    // We may be concurrently trying both execute and force a query.
    // Ensure that only one of them runs the query.
    if let Some((_, index)) = query.query_cache(qcx).lookup(&key) {
        qcx.tcx.prof.query_cache_hit(index.into());
        return;
    }

    debug_assert!(!query.anon());

    ensure_sufficient_stack(|| {
        try_execute_query::<C, FLAGS, true>(query, qcx, DUMMY_SP, key, Some(dep_node))
    });
}
