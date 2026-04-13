use std::hash::Hash;
use std::mem::ManuallyDrop;

use rustc_data_structures::hash_table::{Entry, HashTable};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_data_structures::{outline, sharded, sync};
use rustc_errors::FatalError;
use rustc_middle::dep_graph::{DepGraphData, DepNodeKey, SerializedDepNodeIndex};
use rustc_middle::query::{
    ActiveKeyStatus, Cycle, EnsureMode, QueryCache, QueryJob, QueryJobId, QueryKey, QueryLatch,
    QueryMode, QueryState, QueryVTable,
};
use rustc_middle::ty::TyCtxt;
use rustc_middle::verify_ich::incremental_verify_ich;
use rustc_span::{DUMMY_SP, Span};
use tracing::warn;

use crate::dep_graph::{DepNode, DepNodeIndex};
use crate::job::{QueryJobInfo, QueryJobMap, create_cycle_error, find_cycle_in_stack};
use crate::plumbing::{current_query_job, loadable_from_disk, next_job_id, start_query};
use crate::query_impl::for_each_query_vtable;

#[inline]
fn equivalent_key<K: Eq, V>(k: K) -> impl Fn(&(K, V)) -> bool {
    move |x| x.0 == k
}

pub(crate) fn all_inactive<'tcx, K>(state: &QueryState<'tcx, K>) -> bool {
    state.active.lock_shards().all(|shard| shard.is_empty())
}

#[derive(Clone, Copy)]
pub enum CollectActiveJobsKind {
    /// We need the full query job map, and we are willing to wait to obtain the query state
    /// shard lock(s).
    Full,

    /// We need the full query job map, and we shouldn't need to wait to obtain the shard lock(s),
    /// because we are in a place where nothing else could hold the shard lock(s).
    FullNoContention,

    /// We can get by without the full query job map, so we won't bother waiting to obtain the
    /// shard lock(s) if they're not already unlocked.
    PartialAllowed,
}

/// Returns a map of currently active query jobs, collected from all queries.
pub fn collect_active_query_jobs<'tcx>(
    tcx: TyCtxt<'tcx>,
    collect_kind: CollectActiveJobsKind,
) -> QueryJobMap<'tcx> {
    let mut job_map = QueryJobMap::default();

    for_each_query_vtable!(ALL, tcx, |query| {
        collect_active_query_jobs_inner(query, collect_kind, &mut job_map);
    });

    job_map
}

/// Internal plumbing for collecting the set of active jobs for this query.
///
/// Aborts if jobs can't be gathered as specified by `collect_kind`.
fn collect_active_query_jobs_inner<'tcx, C>(
    query: &'tcx QueryVTable<'tcx, C>,
    collect_kind: CollectActiveJobsKind,
    job_map: &mut QueryJobMap<'tcx>,
) where
    C: QueryCache<Key: QueryKey + DynSend + DynSync>,
    QueryVTable<'tcx, C>: DynSync,
{
    let mut collect_shard_jobs = |shard: &HashTable<(C::Key, ActiveKeyStatus<'tcx>)>| {
        for (key, status) in shard.iter() {
            if let ActiveKeyStatus::Started(job) = status {
                // It's fine to call `create_tagged_key` with the shard locked,
                // because it's just a `TaggedQueryKey` variant constructor.
                let tagged_key = (query.create_tagged_key)(*key);
                job_map.insert(job.id, QueryJobInfo { tagged_key, job: job.clone() });
            }
        }
    };

    match collect_kind {
        CollectActiveJobsKind::Full => {
            for shard in query.state.active.lock_shards() {
                collect_shard_jobs(&shard);
            }
        }
        CollectActiveJobsKind::FullNoContention => {
            for shard in query.state.active.try_lock_shards() {
                match shard {
                    Some(shard) => collect_shard_jobs(&shard),
                    None => panic!("Failed to collect active jobs for query `{}`!", query.name),
                }
            }
        }
        CollectActiveJobsKind::PartialAllowed => {
            for shard in query.state.active.try_lock_shards() {
                match shard {
                    Some(shard) => collect_shard_jobs(&shard),
                    None => warn!("Failed to collect active jobs for query `{}`!", query.name),
                }
            }
        }
    }
}

#[cold]
#[inline(never)]
fn handle_cycle<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    cycle: Cycle<'tcx>,
) -> C::Value {
    let error = create_cycle_error(tcx, &cycle);
    (query.handle_cycle_error_fn)(tcx, key, cycle, error)
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

impl<'tcx, K> ActiveJobGuard<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter, and forgets the guard so it won't poison the query.
    fn complete<C>(self, cache: &C, value: C::Value, dep_node_index: DepNodeIndex)
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
            ActiveKeyStatus::Started(job) => job.signal_complete(),
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
fn find_and_handle_cycle<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    try_execute: QueryJobId,
    span: Span,
) -> (C::Value, Option<DepNodeIndex>) {
    // Ensure there were no errors collecting all active jobs.
    // We need the complete map to ensure we find a cycle to break.
    let job_map = collect_active_query_jobs(tcx, CollectActiveJobsKind::FullNoContention);

    let cycle = find_cycle_in_stack(try_execute, job_map, &current_query_job(), span);
    (handle_cycle(query, tcx, key, cycle), None)
}

#[inline(always)]
fn wait_for_query<'tcx, C: QueryCache>(
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
    let result = latch.wait_on(tcx, current, span);

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

/// Shared main part of both [`execute_query_incr_inner`] and [`execute_query_non_incr_inner`].
#[inline(never)]
fn try_execute_query<'tcx, C: QueryCache, const INCR: bool>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    dep_node: Option<DepNode>, // `None` for non-incremental, `Some` for incremental
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

    let current_job_id = current_query_job();

    match state_lock.entry(key_hash, equivalent_key(key), |(k, _)| sharded::make_hash(k)) {
        Entry::Vacant(entry) => {
            // Nothing has computed or is computing the query, so we start a new job and insert it
            // in the state map.
            let id = next_job_id(tcx);
            let job = QueryJob::new(id, span, current_job_id);
            entry.insert((key, ActiveKeyStatus::Started(job)));

            // Drop the lock before we start executing the query.
            drop(state_lock);

            // Set up a guard object that will automatically poison the query if a
            // panic occurs while executing the query (or any intermediate plumbing).
            let job_guard = ActiveJobGuard { state: &query.state, key, key_hash };

            // Delegate to another function to actually execute the query job.
            let (value, dep_node_index) = if INCR {
                execute_job_incr(query, tcx, key, dep_node.unwrap(), id)
            } else {
                execute_job_non_incr(query, tcx, key, id)
            };

            if query.feedable {
                check_feedable_consistency(tcx, query, key, &value);
            }

            // Tell the guard to insert `value` in the cache and remove the status entry from
            // `query.state`.
            job_guard.complete(&query.cache, value, dep_node_index);

            (value, Some(dep_node_index))
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
                        wait_for_query(query, tcx, span, key, key_hash, latch, current_job_id)
                    } else {
                        let id = job.id;
                        drop(state_lock);

                        // If we are single-threaded we know that we have cycle error,
                        // so we just return the error.
                        find_and_handle_cycle(query, tcx, key, id, span)
                    }
                }
                ActiveKeyStatus::Poisoned => FatalError.raise(),
            }
        }
    }
}

#[inline(always)]
fn check_feedable_consistency<'tcx, C: QueryCache>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
    value: &C::Value,
) {
    // We should not compute queries that also got a value via feeding.
    // This can't happen, as query feeding adds the very dependencies to the fed query
    // as its feeding query had. So if the fed query is red, so is its feeder, which will
    // get evaluated first, and re-feed the query.
    let Some((cached_value, _)) = query.cache.lookup(&key) else { return };

    let Some(hash_value_fn) = query.hash_value_fn else {
        panic!(
            "no_hash fed query later has its value computed.\n\
            Remove `no_hash` modifier to allow recomputation.\n\
            The already cached value: {}",
            (query.format_value)(&cached_value)
        );
    };

    let (old_hash, new_hash) = tcx.with_stable_hashing_context(|mut hcx| {
        (hash_value_fn(&mut hcx, &cached_value), hash_value_fn(&mut hcx, value))
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
            formatter(&cached_value),
        );
    }
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

    let prof_timer = tcx.prof.query_provider();
    // Call the query provider.
    let value = start_query(job_id, query.depth_limit, || (query.invoke_provider_fn)(tcx, key));
    let dep_node_index = tcx.dep_graph.next_virtual_depnode_index();
    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    // Sanity: Fingerprint the key and the result to assert they don't contain anything unhashable.
    if cfg!(debug_assertions) {
        let _ = key.to_fingerprint(tcx);
        if let Some(hash_value_fn) = query.hash_value_fn {
            tcx.with_stable_hashing_context(|mut hcx| {
                hash_value_fn(&mut hcx, &value);
            });
        }
    }

    (value, dep_node_index)
}

#[inline(always)]
fn execute_job_incr<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    dep_node: DepNode,
    job_id: QueryJobId,
) -> (C::Value, DepNodeIndex) {
    let dep_graph_data =
        tcx.dep_graph.data().expect("should always be present in incremental mode");

    if !query.eval_always {
        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = start_query(job_id, false, || try {
            let (prev_index, dep_node_index) = dep_graph_data.try_mark_green(tcx, &dep_node)?;
            let value = load_from_disk_or_invoke_provider_green(
                tcx,
                dep_graph_data,
                query,
                key,
                &dep_node,
                prev_index,
                dep_node_index,
            );
            (value, dep_node_index)
        }) {
            return ret;
        }
    }

    let prof_timer = tcx.prof.query_provider();

    let (result, dep_node_index) = start_query(job_id, query.depth_limit, || {
        // Call the query provider.
        dep_graph_data.with_task(
            dep_node,
            tcx,
            || (query.invoke_provider_fn)(tcx, key),
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
    key: C::Key,
    dep_node: &DepNode,
    prev_index: SerializedDepNodeIndex,
    dep_node_index: DepNodeIndex,
) -> C::Value {
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    debug_assert!(dep_graph_data.is_index_green(prev_index));

    // First try to load the result from the on-disk cache. Some things are never cached on disk.
    let try_value = if (query.will_cache_on_disk_for_key_fn)(key) {
        let prof_timer = tcx.prof.incr_cache_loading();
        let value = (query.try_load_from_disk_fn)(tcx, prev_index);
        prof_timer.finish_with_query_invocation_id(dep_node_index.into());
        value
    } else {
        None
    };
    let (value, verify) = match try_value {
        Some(value) => {
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
            // give us some coverage of potential bugs.
            let verify = prev_fingerprint.split().1.as_u64().is_multiple_of(32)
                || tcx.sess.opts.unstable_opts.incremental_verify_ich;

            (value, verify)
        }
        None => {
            // We could not load a result from the on-disk cache, so recompute. The dep-graph for
            // this computation is already in-place, so we can just call the query provider.
            let prof_timer = tcx.prof.query_provider();
            let value = tcx.dep_graph.with_ignore(|| (query.invoke_provider_fn)(tcx, key));
            prof_timer.finish_with_query_invocation_id(dep_node_index.into());

            (value, true)
        }
    };

    if verify {
        // Verify that re-running the query produced a result with the expected hash.
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
    }

    value
}

/// Checks whether a `tcx.ensure_ok()` or `tcx.ensure_done()` query call can
/// return early without actually trying to execute.
///
/// This only makes sense during incremental compilation, because it relies
/// on having the dependency graph (and in some cases a disk-cached value)
/// from the previous incr-comp session.
#[inline(never)]
fn ensure_can_skip_execution<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    dep_node: DepNode,
    ensure_mode: EnsureMode,
) -> bool {
    // Queries with `eval_always` should never skip execution.
    if query.eval_always {
        return false;
    }

    match tcx.dep_graph.try_mark_green(tcx, &dep_node) {
        None => {
            // A None return from `try_mark_green` means that this is either
            // a new dep node or that the dep node has already been marked red.
            // Either way, we can't call `dep_graph.read()` as we don't have the
            // DepNodeIndex. We must invoke the query itself. The performance cost
            // this introduces should be negligible as we'll immediately hit the
            // in-memory cache, or another query down the line will.
            false
        }
        Some((serialized_dep_node_index, dep_node_index)) => {
            tcx.dep_graph.read_index(dep_node_index);
            tcx.prof.query_cache_hit(dep_node_index.into());
            match ensure_mode {
                // In ensure-ok mode, we can skip execution for this key if the
                // node is green. It must have succeeded in the previous
                // session, and therefore would succeed in the current session
                // if executed.
                EnsureMode::Ok => true,

                // In ensure-done mode, we can only skip execution for this key
                // if there's a disk-cached value available to load later if
                // needed, which guarantees the query provider will never run
                // for this key.
                EnsureMode::Done => {
                    (query.will_cache_on_disk_for_key_fn)(key)
                        && loadable_from_disk(tcx, serialized_dep_node_index)
                }
            }
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
    let dep_node = DepNode::construct(tcx, query.dep_kind, &key);

    // Check if query execution can be skipped, for `ensure_ok` or `ensure_done`.
    if let QueryMode::Ensure { ensure_mode } = mode
        && ensure_can_skip_execution(query, tcx, key, dep_node, ensure_mode)
    {
        return None;
    }

    let (result, dep_node_index) = ensure_sufficient_stack(|| {
        try_execute_query::<C, true>(query, tcx, span, key, Some(dep_node))
    });
    if let Some(dep_node_index) = dep_node_index {
        tcx.dep_graph.read_index(dep_node_index)
    }
    Some(result)
}

/// Inner implementation of [`DepKindVTable::force_from_dep_node_fn`][force_fn]
/// for query nodes.
///
/// [force_fn]: rustc_middle::dep_graph::DepKindVTable::force_from_dep_node_fn
pub(crate) fn force_query_dep_node<'tcx, C: QueryCache>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    dep_node: DepNode,
) -> bool {
    let Some(key) = C::Key::try_recover_key(tcx, &dep_node) else {
        // We couldn't recover a key from the node's key fingerprint.
        // Tell the caller that we couldn't force the node.
        return false;
    };

    ensure_sufficient_stack(|| {
        try_execute_query::<C, true>(query, tcx, DUMMY_SP, key, Some(dep_node))
    });

    // We did manage to recover a key and force the node, though it's up to
    // the caller to check whether the node ended up marked red or green.
    true
}
