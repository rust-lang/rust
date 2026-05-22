use rustc_data_structures::cache_entry::{self, EntryInProgress};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_middle::dep_graph::{DepGraphData, DepNodeKey, SerializedDepNodeIndex};
use rustc_middle::query::{EnsureMode, QueryCache, QueryJob, QueryJobRef, QueryMode, QueryVTable};
use rustc_middle::ty::{TyCtxt, tls};
use rustc_middle::verify_ich::incremental_verify_ich;
use rustc_span::{DUMMY_SP, Span};

use crate::dep_graph::{DepNode, DepNodeIndex};
use crate::job::_find_cycle_in_stack;
use crate::plumbing::{loadable_from_disk, start_query};

#[inline]
fn _equivalent_key<K: Eq, V>(k: K) -> impl Fn(&(K, V)) -> bool {
    move |x| x.0 == k
}

#[cold]
#[inline(never)]
fn _find_and_handle_cycle<'a, 'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    try_execute: QueryJobRef<'a, 'tcx>,
    span: Span,
) -> (C::Value, Option<DepNodeIndex>) {
    tls::with_related_context(tcx, |icx| {
        let cycle = _find_cycle_in_stack(try_execute, icx.query, span);
        ((query.handle_cycle_error_fn)(tcx, key, cycle), None)
    })
}

/// Shared main part of both [`execute_query_incr_inner`] and [`execute_query_non_incr_inner`].
#[inline(never)]
fn try_execute_query<'a, 'tcx, C: QueryCache, const INCR: bool>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    dep_node: Option<DepNode>, // `None` for non-incremental, `Some` for incremental
    entry: EntryInProgress<'a, C::Value>,
) -> (&'a C::Value, DepNodeIndex) {
    tls::with_related_context(tcx, |icx| {
        let form_tagged_key = || (query.create_tagged_key)(key);
        let job = QueryJob {
            span,
            parent: icx.query,
            form_tagged_key: &form_tagged_key,
            entry_status: entry.entry().status(),
        };
        // Delegate to another function to actually execute the query job.
        let (value, dep_node_index) = if INCR {
            execute_job_incr(query, tcx, key, dep_node.unwrap(), &job)
        } else {
            execute_job_non_incr(query, tcx, key, &job)
        };

        let value = entry.complete(value, dep_node_index.as_u32(), &tcx.parking_area);

        (value, dep_node_index)
    })
}

// Fast path for when incr. comp. is off.
#[inline(always)]
fn execute_job_non_incr<'a, 'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    job: QueryJobRef<'a, 'tcx>,
) -> (C::Value, DepNodeIndex) {
    debug_assert!(!tcx.dep_graph.is_fully_enabled());

    let prof_timer = tcx.prof.query_provider();
    // Call the query provider.
    let value = start_query(tcx, job, query.depth_limit, || (query.invoke_provider_fn)(tcx, key));
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
fn execute_job_incr<'a, 'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    key: C::Key,
    dep_node: DepNode,
    job: QueryJobRef<'a, 'tcx>,
) -> (C::Value, DepNodeIndex) {
    let dep_graph_data =
        tcx.dep_graph.data().expect("should always be present in incremental mode");

    if !query.eval_always {
        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = start_query(tcx, job, false, || try {
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

    let (result, dep_node_index) = start_query(tcx, job, query.depth_limit, || {
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
pub(super) fn execute_query_non_incr_inner<'a, 'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    mode: QueryMode<'a, C::Value>,
) -> C::Value {
    let entry = match mode {
        QueryMode::Get { entry } => entry,
        QueryMode::Ensure { entry, .. } => match tcx.cache_entry_get_or_start(entry, span) {
            Ok((result, _)) => return *result,
            Err(cache_entry::GetOrStartError::InProgress(entry)) => entry,
            Err(cache_entry::GetOrStartError::Interrupted(cycle)) => {
                return (query.handle_cycle_error_fn)(tcx, key, cycle);
            }
        },
    };
    ensure_sufficient_stack(|| *try_execute_query::<C, false>(query, tcx, span, key, None, entry).0)
}

/// Called by a macro-generated impl of [`QueryVTable::execute_query_fn`],
/// in incremental mode.
#[inline(always)]
pub(super) fn execute_query_incr_inner<'a, 'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
    span: Span,
    key: C::Key,
    mode: QueryMode<'a, C::Value>,
) -> Option<C::Value> {
    let dep_node = DepNode::construct(tcx, query.dep_kind, &key);

    // Check if query execution can be skipped, for `ensure_ok` or `ensure_done`.
    let entry = match mode {
        QueryMode::Ensure { ensure_mode, entry } => {
            if ensure_can_skip_execution(query, tcx, key, dep_node, ensure_mode) {
                return None;
            }
            match tcx.cache_entry_get_or_start(entry, span) {
                Ok((result, dep_node_index)) => {
                    tcx.dep_graph.read_index(dep_node_index);
                    return Some(*result);
                }
                Err(cache_entry::GetOrStartError::InProgress(entry)) => entry,
                Err(cache_entry::GetOrStartError::Interrupted(cycle)) => {
                    return Some((query.handle_cycle_error_fn)(tcx, key, cycle));
                }
            }
        }
        QueryMode::Get { entry } => entry,
    };

    let (result, dep_node_index) = ensure_sufficient_stack(|| {
        try_execute_query::<C, true>(query, tcx, span, key, Some(dep_node), entry)
    });
    tcx.dep_graph.read_index(dep_node_index);
    Some(*result)
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

    let entry = query.cache.lookup(key);
    match tcx.cache_entry_get_or_start(entry, DUMMY_SP) {
        Err(cache_entry::GetOrStartError::InProgress(entry)) => {
            ensure_sufficient_stack(|| {
                try_execute_query::<C, true>(query, tcx, DUMMY_SP, key, Some(dep_node), entry)
            });
        }
        Err(cache_entry::GetOrStartError::Interrupted(cycle)) => {
            (query.handle_cycle_error_fn)(tcx, key, cycle);
        }
        Ok(_) => (),
    }

    // We did manage to recover a key and force the node, though it's up to
    // the caller to check whether the node ended up marked red or green.
    true
}
