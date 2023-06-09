//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::{DepContext, DepKind, DepNode, DepNodeIndex, DepNodeParams};
use crate::dep_graph::{DepGraphData, HasDepContext};
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
#[cfg(parallel_compiler)]
use crate::query::job::QueryLatch;
use crate::query::job::{report_cycle, QueryInfo, QueryJob, QueryJobId, QueryJobInfo};
use crate::query::SerializedDepNodeIndex;
use crate::query::{QueryContext, QueryMap, QuerySideEffects, QueryStackFrame};
use crate::HandleCycleError;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::Lock;
#[cfg(parallel_compiler)]
use rustc_data_structures::{cold_path, sharded::Sharded};
use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed, FatalError};
use rustc_span::{Span, DUMMY_SP};
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use thin_vec::ThinVec;

use super::QueryConfig;

pub struct QueryState<K, D: DepKind> {
    #[cfg(parallel_compiler)]
    active: Sharded<FxHashMap<K, QueryResult<D>>>,
    #[cfg(not(parallel_compiler))]
    active: Lock<FxHashMap<K, QueryResult<D>>>,
}

/// Indicates the state of a query for a given key in a query map.
enum QueryResult<D: DepKind> {
    /// An already executing query. The query job can be used to await for its completion.
    Started(QueryJob<D>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error which will
    /// silently panic.
    Poisoned,
}

impl<K, D> QueryState<K, D>
where
    K: Eq + Hash + Copy + Debug,
    D: DepKind,
{
    pub fn all_inactive(&self) -> bool {
        #[cfg(parallel_compiler)]
        {
            let shards = self.active.lock_shards();
            shards.iter().all(|shard| shard.is_empty())
        }
        #[cfg(not(parallel_compiler))]
        {
            self.active.lock().is_empty()
        }
    }

    pub fn try_collect_active_jobs<Qcx: Copy>(
        &self,
        qcx: Qcx,
        make_query: fn(Qcx, K) -> QueryStackFrame<D>,
        jobs: &mut QueryMap<D>,
    ) -> Option<()> {
        let mut active = Vec::new();

        #[cfg(parallel_compiler)]
        {
            // We use try_lock_shards here since we are called from the
            // deadlock handler, and this shouldn't be locked.
            let shards = self.active.try_lock_shards()?;
            for shard in shards.iter() {
                for (k, v) in shard.iter() {
                    if let QueryResult::Started(ref job) = *v {
                        active.push((*k, job.clone()));
                    }
                }
            }
        }
        #[cfg(not(parallel_compiler))]
        {
            // We use try_lock here since we are called from the
            // deadlock handler, and this shouldn't be locked.
            // (FIXME: Is this relevant for non-parallel compilers? It doesn't
            // really hurt much.)
            for (k, v) in self.active.try_lock()?.iter() {
                if let QueryResult::Started(ref job) = *v {
                    active.push((*k, job.clone()));
                }
            }
        }

        // Call `make_query` while we're not holding a `self.active` lock as `make_query` may call
        // queries leading to a deadlock.
        for (key, job) in active {
            let query = make_query(qcx, key);
            jobs.insert(job.id, QueryJobInfo { query, job });
        }

        Some(())
    }
}

impl<K, D: DepKind> Default for QueryState<K, D> {
    fn default() -> QueryState<K, D> {
        QueryState { active: Default::default() }
    }
}

/// A type representing the responsibility to execute the job in the `job` field.
/// This will poison the relevant query if dropped.
struct JobOwner<'tcx, K, D: DepKind>
where
    K: Eq + Hash + Copy,
{
    state: &'tcx QueryState<K, D>,
    key: K,
}

#[cold]
#[inline(never)]
fn mk_cycle<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    cycle_error: CycleError<Qcx::DepKind>,
    handler: HandleCycleError,
) -> Q::Value
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    let error = report_cycle(qcx.dep_context().sess(), &cycle_error);
    handle_cycle_error(query, qcx, &cycle_error, error, handler)
}

fn handle_cycle_error<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    cycle_error: &CycleError<Qcx::DepKind>,
    mut error: DiagnosticBuilder<'_, ErrorGuaranteed>,
    handler: HandleCycleError,
) -> Q::Value
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    use HandleCycleError::*;
    match handler {
        Error => {
            error.emit();
            query.value_from_cycle_error(*qcx.dep_context(), &cycle_error.cycle)
        }
        Fatal => {
            error.emit();
            qcx.dep_context().sess().abort_if_errors();
            unreachable!()
        }
        DelayBug => {
            error.delay_as_bug();
            query.value_from_cycle_error(*qcx.dep_context(), &cycle_error.cycle)
        }
    }
}

impl<'tcx, K, D: DepKind> JobOwner<'tcx, K, D>
where
    K: Eq + Hash + Copy,
{
    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter and forgets the JobOwner, so it won't poison the query
    fn complete<C>(self, cache: &C, result: C::Value, dep_node_index: DepNodeIndex)
    where
        C: QueryCache<Key = K>,
    {
        let key = self.key;
        let state = self.state;

        // Forget ourself so our destructor won't poison the query
        mem::forget(self);

        // Mark as complete before we remove the job from the active state
        // so no other thread can re-execute this query.
        cache.complete(key, result, dep_node_index);

        let job = {
            #[cfg(parallel_compiler)]
            let mut lock = state.active.get_shard_by_value(&key).lock();
            #[cfg(not(parallel_compiler))]
            let mut lock = state.active.lock();
            match lock.remove(&key).unwrap() {
                QueryResult::Started(job) => job,
                QueryResult::Poisoned => panic!(),
            }
        };

        job.signal_complete();
    }
}

impl<'tcx, K, D> Drop for JobOwner<'tcx, K, D>
where
    K: Eq + Hash + Copy,
    D: DepKind,
{
    #[inline(never)]
    #[cold]
    fn drop(&mut self) {
        // Poison the query so jobs waiting on it panic.
        let state = self.state;
        let job = {
            #[cfg(parallel_compiler)]
            let mut shard = state.active.get_shard_by_value(&self.key).lock();
            #[cfg(not(parallel_compiler))]
            let mut shard = state.active.lock();
            let job = match shard.remove(&self.key).unwrap() {
                QueryResult::Started(job) => job,
                QueryResult::Poisoned => panic!(),
            };
            shard.insert(self.key, QueryResult::Poisoned);
            job
        };
        // Also signal the completion of the job, so waiters
        // will continue execution.
        job.signal_complete();
    }
}

#[derive(Clone)]
pub(crate) struct CycleError<D: DepKind> {
    /// The query and related span that uses the cycle.
    pub usage: Option<(Span, QueryStackFrame<D>)>,
    pub cycle: Vec<QueryInfo<D>>,
}

/// Checks if the query is already computed and in the cache.
/// It returns the shard index and a lock guard to the shard,
/// which will be used if the query is not in the cache and we need
/// to compute it.
#[inline(always)]
pub fn try_get_cached<Tcx, C>(tcx: Tcx, cache: &C, key: &C::Key) -> Option<C::Value>
where
    C: QueryCache,
    Tcx: DepContext,
{
    match cache.lookup(&key) {
        Some((value, index)) => {
            tcx.profiler().query_cache_hit(index.into());
            tcx.dep_graph().read_index(index);
            Some(value)
        }
        None => None,
    }
}

#[cold]
#[inline(never)]
#[cfg(not(parallel_compiler))]
fn cycle_error<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    try_execute: QueryJobId,
    span: Span,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    let error = try_execute.find_cycle_in_stack(
        qcx.try_collect_active_jobs().unwrap(),
        &qcx.current_query_job(),
        span,
    );
    (mk_cycle(query, qcx, error, query.handle_cycle_error()), None)
}

#[inline(always)]
#[cfg(parallel_compiler)]
fn wait_for_query<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    span: Span,
    key: Q::Key,
    latch: QueryLatch<Qcx::DepKind>,
    current: Option<QueryJobId>,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    // For parallel queries, we'll block and wait until the query running
    // in another thread has completed. Record how long we wait in the
    // self-profiler.
    let query_blocked_prof_timer = qcx.dep_context().profiler().query_blocked();

    // With parallel queries we might just have to wait on some other
    // thread.
    let result = latch.wait_on(current, span);

    match result {
        Ok(()) => {
            let Some((v, index)) = query.query_cache(qcx).lookup(&key) else {
                cold_path(|| panic!("value must be in cache after waiting"))
            };

            qcx.dep_context().profiler().query_cache_hit(index.into());
            query_blocked_prof_timer.finish_with_query_invocation_id(index.into());

            (v, Some(index))
        }
        Err(cycle) => (mk_cycle(query, qcx, cycle, query.handle_cycle_error()), None),
    }
}

#[inline(never)]
fn try_execute_query<Q, Qcx, const INCR: bool>(
    query: Q,
    qcx: Qcx,
    span: Span,
    key: Q::Key,
    dep_node: Option<DepNode<Qcx::DepKind>>,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    let state = query.query_state(qcx);
    #[cfg(parallel_compiler)]
    let mut state_lock = state.active.get_shard_by_value(&key).lock();
    #[cfg(not(parallel_compiler))]
    let mut state_lock = state.active.lock();

    // For the parallel compiler we need to check both the query cache and query state structures
    // while holding the state lock to ensure that 1) the query has not yet completed and 2) the
    // query is not still executing. Without checking the query cache here, we can end up
    // re-executing the query since `try_start` only checks that the query is not currently
    // executing, but another thread may have already completed the query and stores it result
    // in the query cache.
    if cfg!(parallel_compiler) && qcx.dep_context().sess().threads() > 1 {
        if let Some((value, index)) = query.query_cache(qcx).lookup(&key) {
            qcx.dep_context().profiler().query_cache_hit(index.into());
            return (value, Some(index));
        }
    }

    let current_job_id = qcx.current_query_job();

    match state_lock.entry(key) {
        Entry::Vacant(entry) => {
            // Nothing has computed or is computing the query, so we start a new job and insert it in the
            // state map.
            let id = qcx.next_job_id();
            let job = QueryJob::new(id, span, current_job_id);
            entry.insert(QueryResult::Started(job));

            // Drop the lock before we start executing the query
            drop(state_lock);

            execute_job::<_, _, INCR>(query, qcx, state, key, id, dep_node)
        }
        Entry::Occupied(mut entry) => {
            match entry.get_mut() {
                #[cfg(not(parallel_compiler))]
                QueryResult::Started(job) => {
                    let id = job.id;
                    drop(state_lock);

                    // If we are single-threaded we know that we have cycle error,
                    // so we just return the error.
                    cycle_error(query, qcx, id, span)
                }
                #[cfg(parallel_compiler)]
                QueryResult::Started(job) => {
                    // Get the latch out
                    let latch = job.latch();
                    drop(state_lock);

                    wait_for_query(query, qcx, span, key, latch, current_job_id)
                }
                QueryResult::Poisoned => FatalError.raise(),
            }
        }
    }
}

#[inline(always)]
fn execute_job<Q, Qcx, const INCR: bool>(
    query: Q,
    qcx: Qcx,
    state: &QueryState<Q::Key, Qcx::DepKind>,
    key: Q::Key,
    id: QueryJobId,
    dep_node: Option<DepNode<Qcx::DepKind>>,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    // Use `JobOwner` so the query will be poisoned if executing it panics.
    let job_owner = JobOwner { state, key };

    debug_assert_eq!(qcx.dep_context().dep_graph().is_fully_enabled(), INCR);

    let (result, dep_node_index) = if INCR {
        execute_job_incr(
            query,
            qcx,
            qcx.dep_context().dep_graph().data().unwrap(),
            key,
            dep_node,
            id,
        )
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
                // results is tainted by errors. In this case, delay a bug to
                // ensure compilation is doomed.
                qcx.dep_context().sess().delay_span_bug(
                    DUMMY_SP,
                    format!(
                        "Computed query value for {:?}({:?}) is inconsistent with fed value,\n\
                        computed={:#?}\nfed={:#?}",
                        query.dep_kind(),
                        key,
                        formatter(&result),
                        formatter(&cached_result),
                    ),
                );
            }
        }
    }
    job_owner.complete(cache, result, dep_node_index);

    (result, Some(dep_node_index))
}

// Fast path for when incr. comp. is off.
#[inline(always)]
fn execute_job_non_incr<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    key: Q::Key,
    job_id: QueryJobId,
) -> (Q::Value, DepNodeIndex)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    debug_assert!(!qcx.dep_context().dep_graph().is_fully_enabled());

    // Fingerprint the key, just to assert that it doesn't
    // have anything we don't consider hashable
    if cfg!(debug_assertions) {
        let _ = key.to_fingerprint(*qcx.dep_context());
    }

    let prof_timer = qcx.dep_context().profiler().query_provider();
    let result = qcx.start_query(job_id, query.depth_limit(), None, || query.compute(qcx, key));
    let dep_node_index = qcx.dep_context().dep_graph().next_virtual_depnode_index();
    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    // Similarly, fingerprint the result to assert that
    // it doesn't have anything not considered hashable.
    if cfg!(debug_assertions) && let Some(hash_result) = query.hash_result() {
        qcx.dep_context().with_stable_hashing_context(|mut hcx| {
            hash_result(&mut hcx, &result);
        });
    }

    (result, dep_node_index)
}

#[inline(always)]
fn execute_job_incr<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    dep_graph_data: &DepGraphData<Qcx::DepKind>,
    key: Q::Key,
    mut dep_node_opt: Option<DepNode<Qcx::DepKind>>,
    job_id: QueryJobId,
) -> (Q::Value, DepNodeIndex)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    if !query.anon() && !query.eval_always() {
        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node =
            dep_node_opt.get_or_insert_with(|| query.construct_dep_node(*qcx.dep_context(), &key));

        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = qcx.start_query(job_id, false, None, || {
            try_load_from_disk_and_cache_in_memory(query, dep_graph_data, qcx, &key, &dep_node)
        }) {
            return ret;
        }
    }

    let prof_timer = qcx.dep_context().profiler().query_provider();
    let diagnostics = Lock::new(ThinVec::new());

    let (result, dep_node_index) =
        qcx.start_query(job_id, query.depth_limit(), Some(&diagnostics), || {
            if query.anon() {
                return dep_graph_data.with_anon_task(*qcx.dep_context(), query.dep_kind(), || {
                    query.compute(qcx, key)
                });
            }

            // `to_dep_node` is expensive for some `DepKind`s.
            let dep_node =
                dep_node_opt.unwrap_or_else(|| query.construct_dep_node(*qcx.dep_context(), &key));

            dep_graph_data.with_task(
                dep_node,
                (qcx, query),
                key,
                |(qcx, query), key| query.compute(qcx, key),
                query.hash_result(),
            )
        });

    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    let diagnostics = diagnostics.into_inner();
    let side_effects = QuerySideEffects { diagnostics };

    if std::intrinsics::unlikely(!side_effects.is_empty()) {
        if query.anon() {
            qcx.store_side_effects_for_anon_node(dep_node_index, side_effects);
        } else {
            qcx.store_side_effects(dep_node_index, side_effects);
        }
    }

    (result, dep_node_index)
}

#[inline(always)]
fn try_load_from_disk_and_cache_in_memory<Q, Qcx>(
    query: Q,
    dep_graph_data: &DepGraphData<Qcx::DepKind>,
    qcx: Qcx,
    key: &Q::Key,
    dep_node: &DepNode<Qcx::DepKind>,
) -> Option<(Q::Value, DepNodeIndex)>
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    let (prev_dep_node_index, dep_node_index) = dep_graph_data.try_mark_green(qcx, &dep_node)?;

    debug_assert!(dep_graph_data.is_index_green(prev_dep_node_index));

    // First we try to load the result from the on-disk cache.
    // Some things are never cached on disk.
    if let Some(result) = query.try_load_from_disk(qcx, key, prev_dep_node_index, dep_node_index) {
        if std::intrinsics::unlikely(qcx.dep_context().sess().opts.unstable_opts.query_dep_graph) {
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
        let try_verify = prev_fingerprint.split().1.as_u64() % 32 == 0;
        if std::intrinsics::unlikely(
            try_verify || qcx.dep_context().sess().opts.unstable_opts.incremental_verify_ich,
        ) {
            incremental_verify_ich(
                *qcx.dep_context(),
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
        !query.cache_on_disk(*qcx.dep_context(), key)
            || !qcx.dep_context().fingerprint_style(dep_node.kind).reconstructible(),
        "missing on-disk cache entry for {dep_node:?}"
    );

    // Sanity check for the logic in `ensure`: if the node is green and the result loadable,
    // we should actually be able to load it.
    debug_assert!(
        !query.loadable_from_disk(qcx, &key, prev_dep_node_index),
        "missing on-disk cache entry for loadable {dep_node:?}"
    );

    // We could not load a result from the on-disk cache, so
    // recompute.
    let prof_timer = qcx.dep_context().profiler().query_provider();

    // The dep-graph for this computation is already in-place.
    let result = qcx.dep_context().dep_graph().with_ignore(|| query.compute(qcx, *key));

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
        *qcx.dep_context(),
        dep_graph_data,
        &result,
        prev_dep_node_index,
        query.hash_result(),
        query.format_value(),
    );

    Some((result, dep_node_index))
}

#[inline]
#[instrument(skip(tcx, dep_graph_data, result, hash_result, format_value), level = "debug")]
pub(crate) fn incremental_verify_ich<Tcx, V>(
    tcx: Tcx,
    dep_graph_data: &DepGraphData<Tcx::DepKind>,
    result: &V,
    prev_index: SerializedDepNodeIndex,
    hash_result: Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>,
    format_value: fn(&V) -> String,
) where
    Tcx: DepContext,
{
    if !dep_graph_data.is_index_green(prev_index) {
        incremental_verify_ich_not_green(tcx, prev_index)
    }

    let new_hash = hash_result.map_or(Fingerprint::ZERO, |f| {
        tcx.with_stable_hashing_context(|mut hcx| f(&mut hcx, result))
    });

    let old_hash = dep_graph_data.prev_fingerprint_of(prev_index);

    if new_hash != old_hash {
        incremental_verify_ich_failed(tcx, prev_index, &|| format_value(&result));
    }
}

#[cold]
#[inline(never)]
fn incremental_verify_ich_not_green<Tcx>(tcx: Tcx, prev_index: SerializedDepNodeIndex)
where
    Tcx: DepContext,
{
    panic!(
        "fingerprint for green query instance not loaded from cache: {:?}",
        tcx.dep_graph().data().unwrap().prev_node_of(prev_index)
    )
}

// Note that this is marked #[cold] and intentionally takes `dyn Debug` for `result`,
// as we want to avoid generating a bunch of different implementations for LLVM to
// chew on (and filling up the final binary, too).
#[cold]
#[inline(never)]
fn incremental_verify_ich_failed<Tcx>(
    tcx: Tcx,
    prev_index: SerializedDepNodeIndex,
    result: &dyn Fn() -> String,
) where
    Tcx: DepContext,
{
    // When we emit an error message and panic, we try to debug-print the `DepNode`
    // and query result. Unfortunately, this can cause us to run additional queries,
    // which may result in another fingerprint mismatch while we're in the middle
    // of processing this one. To avoid a double-panic (which kills the process
    // before we can print out the query static), we print out a terse
    // but 'safe' message if we detect a reentrant call to this method.
    thread_local! {
        static INSIDE_VERIFY_PANIC: Cell<bool> = const { Cell::new(false) };
    };

    let old_in_panic = INSIDE_VERIFY_PANIC.with(|in_panic| in_panic.replace(true));

    if old_in_panic {
        tcx.sess().emit_err(crate::error::Reentrant);
    } else {
        let run_cmd = if let Some(crate_name) = &tcx.sess().opts.crate_name {
            format!("`cargo clean -p {crate_name}` or `cargo clean`")
        } else {
            "`cargo clean`".to_string()
        };

        let dep_node = tcx.dep_graph().data().unwrap().prev_node_of(prev_index);
        tcx.sess().emit_err(crate::error::IncrementCompilation {
            run_cmd,
            dep_node: format!("{dep_node:?}"),
        });
        panic!("Found unstable fingerprints for {dep_node:?}: {}", result());
    }

    INSIDE_VERIFY_PANIC.with(|in_panic| in_panic.set(old_in_panic));
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
fn ensure_must_run<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    key: &Q::Key,
    check_cache: bool,
) -> (bool, Option<DepNode<Qcx::DepKind>>)
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    if query.eval_always() {
        return (true, None);
    }

    // Ensuring an anonymous query makes no sense
    assert!(!query.anon());

    let dep_node = query.construct_dep_node(*qcx.dep_context(), key);

    let dep_graph = qcx.dep_context().dep_graph();
    let serialized_dep_node_index = match dep_graph.try_mark_green(qcx, &dep_node) {
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
            qcx.dep_context().profiler().query_cache_hit(dep_node_index.into());
            serialized_dep_node_index
        }
    };

    // We do not need the value at all, so do not check the cache.
    if !check_cache {
        return (false, None);
    }

    let loadable = query.loadable_from_disk(qcx, key, serialized_dep_node_index);
    (!loadable, Some(dep_node))
}

#[derive(Debug)]
pub enum QueryMode {
    Get,
    Ensure { check_cache: bool },
}

#[inline(always)]
pub fn get_query_non_incr<Q, Qcx>(query: Q, qcx: Qcx, span: Span, key: Q::Key) -> Q::Value
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    debug_assert!(!qcx.dep_context().dep_graph().is_fully_enabled());

    ensure_sufficient_stack(|| try_execute_query::<Q, Qcx, false>(query, qcx, span, key, None).0)
}

#[inline(always)]
pub fn get_query_incr<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    span: Span,
    key: Q::Key,
    mode: QueryMode,
) -> Option<Q::Value>
where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    debug_assert!(qcx.dep_context().dep_graph().is_fully_enabled());

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
        try_execute_query::<_, _, true>(query, qcx, span, key, dep_node)
    });
    if let Some(dep_node_index) = dep_node_index {
        qcx.dep_context().dep_graph().read_index(dep_node_index)
    }
    Some(result)
}

pub fn force_query<Q, Qcx>(
    query: Q,
    qcx: Qcx,
    key: Q::Key,
    dep_node: DepNode<<Qcx as HasDepContext>::DepKind>,
) where
    Q: QueryConfig<Qcx>,
    Qcx: QueryContext,
{
    // We may be concurrently trying both execute and force a query.
    // Ensure that only one of them runs the query.
    if let Some((_, index)) = query.query_cache(qcx).lookup(&key) {
        qcx.dep_context().profiler().query_cache_hit(index.into());
        return;
    }

    debug_assert!(!query.anon());

    ensure_sufficient_stack(|| {
        try_execute_query::<_, _, true>(query, qcx, DUMMY_SP, key, Some(dep_node))
    });
}
