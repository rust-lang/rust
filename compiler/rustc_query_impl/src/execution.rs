use std::hash::Hash;
use std::mem;

use rustc_data_structures::hash_table::{Entry, HashTable};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_data_structures::{outline, sharded, sync};
use rustc_errors::{Diag, FatalError, StashKey};
use rustc_middle::dep_graph::{DepGraphData, DepNodeKey, SerializedDepNodeIndex};
use rustc_middle::query::plumbing::QueryVTable;
use rustc_middle::query::{
    ActiveKeyStatus, CycleError, CycleErrorHandling, EnsureMode, QueryCache, QueryJob, QueryJobId,
    QueryKey, QueryLatch, QueryMode, QueryState,
};
use rustc_middle::ty::TyCtxt;
use rustc_middle::verify_ich::incremental_verify_ich;
use rustc_span::{DUMMY_SP, Span};

use crate::collect_active_jobs_from_all_queries;
use crate::dep_graph::{DepNode, DepNodeIndex};
use crate::job::{QueryJobInfo, QueryJobMap, find_cycle_in_stack, report_cycle};
use crate::plumbing::{current_query_job, next_job_id, start_query};

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
/// Should only be called from `collect_active_jobs_from_all_queries`.
///
/// (We arbitrarily use the word "gather" when collecting the jobs for
/// each individual query, so that we have distinct function names to
/// grep for.)
pub(crate) fn gather_active_jobs<'tcx, C>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    require_complete: bool,
    job_map_out: &mut QueryJobMap<'tcx>, // Out-param; job info is gathered into this map
) -> Option<()>
where
    C: QueryCache<Key: QueryKey + DynSend + DynSync>,
    QueryVTable<'tcx, C>: DynSync,
{
    let mut active = Vec::new();

    // Helper to gather active jobs from a single shard.
    let mut gather_shard_jobs = |shard: &HashTable<(C::Key, ActiveKeyStatus<'tcx>)>| {
        for (k, v) in shard.iter() {
            if let ActiveKeyStatus::Started(ref job) = *v {
                active.push((*k, job.clone()));
            }
        }
    };

    // Lock shards and gather jobs from each shard.
    if require_complete {
        for shard in query.state.active.lock_shards() {
            gather_shard_jobs(&shard);
        }
    } else {
        // We use try_lock_shards here since we are called from the
        // deadlock handler, and this shouldn't be locked.
        for shard in query.state.active.try_lock_shards() {
            // This can be called during unwinding, and the function has a `try_`-prefix, so
            // don't `unwrap()` here, just manually check for `None` and do best-effort error
            // reporting.
            match shard {
                None => {
                    tracing::warn!(
                        "Failed to collect active jobs for query with name `{}`!",
                        query.name
                    );
                    return None;
                }
                Some(shard) => gather_shard_jobs(&shard),
            }
        }
    }

    // Call `make_frame` while we're not holding a `state.active` lock as `make_frame` may call
    // queries leading to a deadlock.
    for (key, job) in active {
        let frame = crate::plumbing::create_deferred_query_stack_frame(tcx, query, key);
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
fn mk_cycle<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    cycle_error: CycleError,
) -> C::Value {
    let error = report_cycle(tcx.sess, &cycle_error);
    handle_cycle_error(query, tcx, &cycle_error, error)
}

fn handle_cycle_error<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    cycle_error: &CycleError,
    error: Diag<'_>,
) -> C::Value {
    match query.cycle_error_handling {
        CycleErrorHandling::Error => {
            let guar = error.emit();
            query.value_from_cycle_error(tcx, cycle_error, guar)
        }
        CycleErrorHandling::Fatal => {
            error.emit();
            tcx.dcx().abort_if_errors();
            unreachable!()
        }
        CycleErrorHandling::DelayBug => {
            let guar = error.delay_as_bug();
            query.value_from_cycle_error(tcx, cycle_error, guar)
        }
        CycleErrorHandling::Stash => {
            let guar = if let Some(root) = cycle_error.cycle.first()
                && let Some(span) = root.frame.info.span
            {
                error.stash(span, StashKey::Cycle).unwrap()
            } else {
                error.emit()
            };
            query.value_from_cycle_error(tcx, cycle_error, guar)
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
fn cycle_error<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    try_execute: QueryJobId,
    span: Span,
) -> (C::Value, Option<DepNodeIndex>) {
    // Ensure there was no errors collecting all active jobs.
    // We need the complete map to ensure we find a cycle to break.
    let job_map = collect_active_jobs_from_all_queries(tcx, false)
        .ok()
        .expect("failed to collect active queries");

    let error = find_cycle_in_stack(try_execute, job_map, &current_query_job(tcx), span);
    (mk_cycle(query, tcx, error.lift()), None)
}

#[inline(always)]
fn wait_for_query<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    latch: QueryLatch<'tcx>,
    current: Option<QueryJobId>,
) -> (C::Value, Option<DepNodeIndex>) {
    // For parallel queries, we'll block and wait until the query running
    // in another thread has completed. Record how long we wait in the
    // self-profiler.
    let query_blocked_prof_timer = tcx.prof.query_blocked();

    // With parallel queries we might just have to wait on some other
    // thread.
    let result = latch.wait_on(tcx, current, span);

    match result {
        Ok(()) => {
            let Some((v, index)) = query.cache.lookup(&key) else {
                outline(|| {
                    // We didn't find the query result in the query cache. Check if it was
                    // poisoned due to a panic instead.
                    let key_hash = sharded::make_hash(&key);
                    let shard = query.state.active.lock_shard_by_hash(key_hash);
                    match shard.find(key_hash, equivalent_key(&key)) {
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
        Err(cycle) => (mk_cycle(query, tcx, cycle.lift()), None),
    }
}

/// Shared main part of both [`execute_query_incr_inner`] and [`execute_query_non_incr_inner`].
#[inline(never)]
fn try_execute_query<'tcx, C: QueryCache, const INCR: bool>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    // If present, some previous step has already created a `DepNode` for this
    // query+key, which we should reuse instead of creating a new one.
    dep_node: Option<DepNode>,
) -> (C::Value, Option<DepNodeIndex>) {
    let key_hash = sharded::make_hash(&key);
    let mut state_lock = query.state.active.lock_shard_by_hash(key_hash);

    // For the parallel compiler we need to check both the query cache and query state structures
    // while holding the state lock to ensure that 1) the query has not yet completed and 2) the
    // query is not still executing. Without checking the query cache here, we can end up
    // re-executing the query since `try_start` only checks that the query is not currently
    // executing, but another thread may have already completed the query and stores it result
    // in the query cache.
    if tcx.sess.threads() > 1 {
        if let Some((value, index)) = query.cache.lookup(&key) {
            tcx.prof.query_cache_hit(index.into());
            return (value, Some(index));
        }
    }

    let current_job_id = current_query_job(tcx);

    match state_lock.entry(key_hash, equivalent_key(&key), |(k, _)| sharded::make_hash(k)) {
        Entry::Vacant(entry) => {
            // Nothing has computed or is computing the query, so we start a new job and insert it in the
            // state map.
            let id = next_job_id(tcx);
            let job = QueryJob::new(id, span, current_job_id);
            entry.insert((key, ActiveKeyStatus::Started(job)));

            // Drop the lock before we start executing the query
            drop(state_lock);

            execute_job::<C, INCR>(query, tcx, key, key_hash, id, dep_node)
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
                        return wait_for_query(query, tcx, span, key, latch, current_job_id);
                    }

                    let id = job.id;
                    drop(state_lock);

                    // If we are single-threaded we know that we have cycle error,
                    // so we just return the error.
                    cycle_error(query, tcx, id, span)
                }
                ActiveKeyStatus::Poisoned => FatalError.raise(),
            }
        }
    }
}

#[inline(always)]
fn execute_job<'tcx, C: QueryCache, const INCR: bool>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    key_hash: u64,
    id: QueryJobId,
    dep_node: Option<DepNode>,
) -> (C::Value, Option<DepNodeIndex>) {
    // Set up a guard object that will automatically poison the query if a
    // panic occurs while executing the query (or any intermediate plumbing).
    let job_guard = ActiveJobGuard { state: &query.state, key, key_hash };

    debug_assert_eq!(tcx.dep_graph.is_fully_enabled(), INCR);

    // Delegate to another function to actually execute the query job.
    let (value, dep_node_index) = if INCR {
        execute_job_incr(query, tcx, key, dep_node, id)
    } else {
        execute_job_non_incr(query, tcx, key, id)
    };

    let cache = &query.cache;
    if query.feedable {
        // We should not compute queries that also got a value via feeding.
        // This can't happen, as query feeding adds the very dependencies to the fed query
        // as its feeding query had. So if the fed query is red, so is its feeder, which will
        // get evaluated first, and re-feed the query.
        if let Some((cached_value, _)) = cache.lookup(&key) {
            let Some(hash_value_fn) = query.hash_value_fn else {
                panic!(
                    "no_hash fed query later has its value computed.\n\
                    Remove `no_hash` modifier to allow recomputation.\n\
                    The already cached value: {}",
                    (query.format_value)(&cached_value)
                );
            };

            let (old_hash, new_hash) = tcx.with_stable_hashing_context(|mut hcx| {
                (hash_value_fn(&mut hcx, &cached_value), hash_value_fn(&mut hcx, &value))
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
                    formatter(&value),
                    formatter(&cached_value),
                );
            }
        }
    }

    // Tell the guard to perform completion bookkeeping, and also to not poison the query.
    job_guard.complete(cache, value, dep_node_index);

    (value, Some(dep_node_index))
}

// Fast path for when incr. comp. is off.
#[inline(always)]
fn execute_job_non_incr<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    job_id: QueryJobId,
) -> (C::Value, DepNodeIndex) {
    debug_assert!(!tcx.dep_graph.is_fully_enabled());

    // Fingerprint the key, just to assert that it doesn't
    // have anything we don't consider hashable
    if cfg!(debug_assertions) {
        let _ = key.to_fingerprint(tcx);
    }

    let prof_timer = tcx.prof.query_provider();
    // Call the query provider.
    let value =
        start_query(tcx, job_id, query.depth_limit, || (query.invoke_provider_fn)(tcx, key));
    let dep_node_index = tcx.dep_graph.next_virtual_depnode_index();
    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    // Similarly, fingerprint the result to assert that
    // it doesn't have anything not considered hashable.
    if cfg!(debug_assertions)
        && let Some(hash_value_fn) = query.hash_value_fn
    {
        tcx.with_stable_hashing_context(|mut hcx| {
            hash_value_fn(&mut hcx, &value);
        });
    }

    (value, dep_node_index)
}

#[inline(always)]
fn execute_job_incr<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    mut dep_node_opt: Option<DepNode>,
    job_id: QueryJobId,
) -> (C::Value, DepNodeIndex) {
    let dep_graph_data =
        tcx.dep_graph.data().expect("should always be present in incremental mode");

    if !query.anon && !query.eval_always {
        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node = dep_node_opt.get_or_insert_with(|| query.construct_dep_node(tcx, &key));

        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = start_query(tcx, job_id, false, || try {
            let (prev_index, dep_node_index) = dep_graph_data.try_mark_green(tcx, dep_node)?;
            let value = load_from_disk_or_invoke_provider_green(
                tcx,
                dep_graph_data,
                query,
                &key,
                dep_node,
                prev_index,
                dep_node_index,
            );
            (value, dep_node_index)
        }) {
            return ret;
        }
    }

    let prof_timer = tcx.prof.query_provider();

    let (result, dep_node_index) = start_query(tcx, job_id, query.depth_limit, || {
        if query.anon {
            // Call the query provider inside an anon task.
            return dep_graph_data.with_anon_task_inner(tcx, query.dep_kind, || {
                (query.invoke_provider_fn)(tcx, key)
            });
        }

        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node = dep_node_opt.unwrap_or_else(|| query.construct_dep_node(tcx, &key));

        // Call the query provider.
        dep_graph_data.with_task(
            dep_node,
            tcx,
            (query, key),
            |tcx, (query, key)| (query.invoke_provider_fn)(tcx, key),
            query.hash_value_fn,
        )
    });

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    (result, dep_node_index)
}

/// Given that the dep node for this query+key is green, obtain a value for it
/// by loading one from disk if possible, or by invoking its query provider if
/// necessary.
#[inline(always)]
fn load_from_disk_or_invoke_provider_green<'tcx, C: QueryCache>(
    tcx: TyCtxt<'tcx>,
    dep_graph_data: &DepGraphData,
    query: &'tcx QueryVTable<'tcx, C>,
    key: &C::Key,
    dep_node: &DepNode,
    prev_index: SerializedDepNodeIndex,
    dep_node_index: DepNodeIndex,
) -> C::Value {
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    debug_assert!(dep_graph_data.is_index_green(prev_index));

    // First we try to load the result from the on-disk cache.
    // Some things are never cached on disk.
    if let Some(value) = query.try_load_from_disk(tcx, key, prev_index, dep_node_index) {
        if std::intrinsics::unlikely(tcx.sess.opts.unstable_opts.query_dep_graph) {
            dep_graph_data.mark_debug_loaded_from_disk(*dep_node)
        }

        let prev_fingerprint = dep_graph_data.prev_value_fingerprint_of(prev_index);
        // If `-Zincremental-verify-ich` is specified, re-hash results from
        // the cache and make sure that they have the expected fingerprint.
        //
        // If not, we still seek to verify a subset of fingerprints loaded
        // from disk. Re-hashing results is fairly expensive, so we can't
        // currently afford to verify every hash. This subset should still
        // give us some coverage of potential bugs though.
        let try_verify = prev_fingerprint.split().1.as_u64().is_multiple_of(32);
        if std::intrinsics::unlikely(
            try_verify || tcx.sess.opts.unstable_opts.incremental_verify_ich,
        ) {
            incremental_verify_ich(
                tcx,
                dep_graph_data,
                &value,
                prev_index,
                query.hash_value_fn,
                query.format_value,
            );
        }

        return value;
    }

    // We always expect to find a cached result for things that
    // can be forced from `DepNode`.
    debug_assert!(
        !query.will_cache_on_disk_for_key(tcx, key)
            || !tcx.key_fingerprint_style(dep_node.kind).is_maybe_recoverable(),
        "missing on-disk cache entry for {dep_node:?}"
    );

    // Sanity check for the logic in `ensure`: if the node is green and the result loadable,
    // we should actually be able to load it.
    debug_assert!(
        !query.is_loadable_from_disk(tcx, key, prev_index),
        "missing on-disk cache entry for loadable {dep_node:?}"
    );

    // We could not load a result from the on-disk cache, so
    // recompute.
    let prof_timer = tcx.prof.query_provider();

    // The dep-graph for this computation is already in-place.
    // Call the query provider.
    let value = tcx.dep_graph.with_ignore(|| (query.invoke_provider_fn)(tcx, *key));

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
        tcx,
        dep_graph_data,
        &value,
        prev_index,
        query.hash_value_fn,
        query.format_value,
    );

    value
}

/// Return value struct for [`check_if_ensure_can_skip_execution`].
struct EnsureCanSkip {
    /// If true, the current `tcx.ensure_ok()` or `tcx.ensure_done()` query
    /// can return early without actually trying to execute.
    skip_execution: bool,
    /// A dep node that was prepared while checking whether execution can be
    /// skipped, to be reused by execution itself if _not_ skipped.
    dep_node: Option<DepNode>,
}

/// Checks whether a `tcx.ensure_ok()` or `tcx.ensure_done()` query call can
/// return early without actually trying to execute.
///
/// This only makes sense during incremental compilation, because it relies
/// on having the dependency graph (and in some cases a disk-cached value)
/// from the previous incr-comp session.
#[inline(never)]
fn check_if_ensure_can_skip_execution<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: &C::Key,
    ensure_mode: EnsureMode,
) -> EnsureCanSkip {
    // Queries with `eval_always` should never skip execution.
    if query.eval_always {
        return EnsureCanSkip { skip_execution: false, dep_node: None };
    }

    // Ensuring an anonymous query makes no sense
    assert!(!query.anon);

    let dep_node = query.construct_dep_node(tcx, key);

    let dep_graph = &tcx.dep_graph;
    let serialized_dep_node_index = match dep_graph.try_mark_green(tcx, &dep_node) {
        None => {
            // A None return from `try_mark_green` means that this is either
            // a new dep node or that the dep node has already been marked red.
            // Either way, we can't call `dep_graph.read()` as we don't have the
            // DepNodeIndex. We must invoke the query itself. The performance cost
            // this introduces should be negligible as we'll immediately hit the
            // in-memory cache, or another query down the line will.
            return EnsureCanSkip { skip_execution: false, dep_node: Some(dep_node) };
        }
        Some((serialized_dep_node_index, dep_node_index)) => {
            dep_graph.read_index(dep_node_index);
            tcx.prof.query_cache_hit(dep_node_index.into());
            serialized_dep_node_index
        }
    };

    match ensure_mode {
        EnsureMode::Ok => {
            // In ensure-ok mode, we can skip execution for this key if the node
            // is green. It must have succeeded in the previous session, and
            // therefore would succeed in the current session if executed.
            EnsureCanSkip { skip_execution: true, dep_node: None }
        }
        EnsureMode::Done => {
            // In ensure-done mode, we can only skip execution for this key if
            // there's a disk-cached value available to load later if needed,
            // which guarantees the query provider will never run for this key.
            let is_loadable = query.is_loadable_from_disk(tcx, key, serialized_dep_node_index);
            EnsureCanSkip { skip_execution: is_loadable, dep_node: Some(dep_node) }
        }
    }
}

/// Called by a macro-generated impl of [`QueryVTable::execute_query_fn`],
/// in non-incremental mode.
#[inline(always)]
pub(super) fn execute_query_non_incr_inner<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
) -> C::Value {
    debug_assert!(!tcx.dep_graph.is_fully_enabled());

    ensure_sufficient_stack(|| try_execute_query::<C, false>(query, tcx, span, key, None).0)
}

/// Called by a macro-generated impl of [`QueryVTable::execute_query_fn`],
/// in incremental mode.
#[inline(always)]
pub(super) fn execute_query_incr_inner<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    mode: QueryMode,
) -> Option<C::Value> {
    debug_assert!(tcx.dep_graph.is_fully_enabled());

    // Check if query execution can be skipped, for `ensure_ok` or `ensure_done`.
    // This might have the side-effect of creating a suitable DepNode, which
    // we should reuse for execution instead of creating a new one.
    let dep_node: Option<DepNode> = match mode {
        QueryMode::Ensure { ensure_mode } => {
            let EnsureCanSkip { skip_execution, dep_node } =
                check_if_ensure_can_skip_execution(query, tcx, &key, ensure_mode);
            if skip_execution {
                // Return early to skip execution.
                return None;
            }
            dep_node
        }
        QueryMode::Get => None,
    };

    let (result, dep_node_index) =
        ensure_sufficient_stack(|| try_execute_query::<C, true>(query, tcx, span, key, dep_node));
    if let Some(dep_node_index) = dep_node_index {
        tcx.dep_graph.read_index(dep_node_index)
    }
    Some(result)
}

pub(crate) fn force_query<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    dep_node: DepNode,
) {
    // We may be concurrently trying both execute and force a query.
    // Ensure that only one of them runs the query.
    if let Some((_, index)) = query.cache.lookup(&key) {
        tcx.prof.query_cache_hit(index.into());
        return;
    }

    debug_assert!(!query.anon);

    ensure_sufficient_stack(|| {
        try_execute_query::<C, true>(query, tcx, DUMMY_SP, key, Some(dep_node))
    });
}
