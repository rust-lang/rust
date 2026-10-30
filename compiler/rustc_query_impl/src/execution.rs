use rustc_data_structures::hash_table::Entry;
use rustc_data_structures::{sharded, sync};
use rustc_errors::FatalError;
use rustc_middle::dep_graph::{
    DepGraphData, DepNode, DepNodeIndex, DepNodeKey, SerializedDepNodeIndex,
};
use rustc_middle::query::q_impl::execution::{
    ActiveJobGuard, check_feedable_consistency, current_query_job, depth_limit_error,
    equivalent_key, find_and_handle_cycle, next_job_id, wait_for_query,
};
use rustc_middle::query::q_impl::incremental::should_verify_loaded_value;
use rustc_middle::query::{
    ActiveKeyStatus, QueryCache, QueryJob, QueryJobId, QueryLatch, QueryMode, QueryVTable,
};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::verify_ich::incremental_verify_ich;
use rustc_span::{DUMMY_SP, Span};

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
    if tcx.sess.opts.jobs.frontend.is_some() {
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

            if query.feedable
                && let Some((cached_value, _)) = query.cache.lookup(&key)
            {
                check_feedable_consistency(tcx, query, key, &value, &cached_value);
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
                        let latch = job.latch.get_or_insert_with(QueryLatch::new).clone();
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

/// Executes a job by changing the `ImplicitCtxt` to point to the new query job while it executes.
#[inline(always)]
fn start_query<R>(job_id: QueryJobId, depth_limit: bool, compute: impl FnOnce() -> R) -> R {
    tls::with_context(move |icx| {
        if depth_limit && !icx.tcx.recursion_limit().value_within_limit(icx.query_depth) {
            depth_limit_error(icx.tcx, job_id);
        }

        // Update the `ImplicitCtxt` to point to our new query job.
        let icx = ImplicitCtxt {
            query: Some(job_id),
            query_depth: icx.query_depth + if depth_limit { 1 } else { 0 },
            ..*icx
        };

        // Use the `ImplicitCtxt` while we execute the query.
        tls::enter_context(&icx, compute)
    })
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
    let try_value = if query.will_cache_on_disk_for_key(key) {
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

            let verify = should_verify_loaded_value(tcx, dep_graph_data, dep_node.key_fingerprint);

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

/// Checks whether a `tcx.ensure_ok()` query call can
/// return early without actually trying to execute.
///
/// This only makes sense during incremental compilation, because it relies
/// on having the dependency graph (and in some cases a disk-cached value)
/// from the previous incr-comp session.
#[inline(never)]
fn ensure_can_skip_execution<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    dep_node: DepNode,
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
        Some((_, dep_node_index)) => {
            tcx.dep_graph.read_index(dep_node_index);
            tcx.prof.query_cache_hit(dep_node_index.into());

            // We can skip execution for this key if the
            // node is green. It must have succeeded in the previous
            // session, and therefore would succeed in the current session
            // if executed.
            true
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
    try_execute_query::<C, false>(query, tcx, span, key, None).0
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

    // Check if query execution can be skipped, for `ensure_ok`.
    if let QueryMode::EnsureOk = mode
        && ensure_can_skip_execution(query, tcx, dep_node)
    {
        return None;
    }

    let (result, dep_node_index) =
        try_execute_query::<C, true>(query, tcx, span, key, Some(dep_node));
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

    try_execute_query::<C, true>(query, tcx, DUMMY_SP, key, Some(dep_node));

    // We did manage to recover a key and force the node, though it's up to
    // the caller to check whether the node ended up marked red or green.
    true
}
