use std::num::NonZero;

use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::DefId;
use rustc_hir::limit::Limit;
use rustc_index::Idx;
use rustc_middle::bug;
#[expect(unused_imports, reason = "used by doc comments")]
use rustc_middle::dep_graph::DepKindVTable;
use rustc_middle::dep_graph::{DepKind, DepNode, DepNodeIndex, DepNodeKey, SerializedDepNodeIndex};
use rustc_middle::query::erase::{Erasable, Erased};
use rustc_middle::query::on_disk_cache::{
    AbsoluteBytePos, CacheDecoder, CacheEncoder, EncodedDepNodeIndex,
};
use rustc_middle::query::plumbing::QueryVTable;
use rustc_middle::query::{QueryCache, QueryJobId, QueryKey, QueryMode, QueryStackFrame, erase};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::LOCAL_CRATE;

use crate::error::{QueryOverflow, QueryOverflowNote};
use crate::execution::{all_inactive, force_query};
use crate::job::find_dep_kind_root;
use crate::query_impl::for_each_query_vtable;
use crate::{CollectActiveJobsKind, GetQueryVTable, collect_active_jobs_from_all_queries};

fn depth_limit_error<'tcx>(tcx: TyCtxt<'tcx>, job: QueryJobId) {
    let job_map = collect_active_jobs_from_all_queries(tcx, CollectActiveJobsKind::Full);
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

#[inline]
pub(crate) fn next_job_id<'tcx>(tcx: TyCtxt<'tcx>) -> QueryJobId {
    QueryJobId(
        NonZero::new(tcx.query_system.jobs.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            .unwrap(),
    )
}

#[inline]
pub(crate) fn current_query_job() -> Option<QueryJobId> {
    tls::with_context(|icx| icx.query)
}

/// Executes a job by changing the `ImplicitCtxt` to point to the new query job while it executes.
#[inline(always)]
pub(crate) fn start_query<R>(
    job_id: QueryJobId,
    depth_limit: bool,
    compute: impl FnOnce() -> R,
) -> R {
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

pub(crate) fn create_query_stack_frame<'tcx, C>(
    vtable: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
) -> QueryStackFrame<'tcx>
where
    C: QueryCache<Key: QueryKey + DynSend + DynSync>,
    QueryVTable<'tcx, C>: DynSync,
{
    let def_id: Option<DefId> = key.key_as_def_id();
    let def_id_for_ty_in_cycle: Option<DefId> = key.def_id_for_ty_in_cycle();

    QueryStackFrame {
        tagged_key: (vtable.create_tagged_key)(key),
        dep_kind: vtable.dep_kind,
        def_id,
        def_id_for_ty_in_cycle,
    }
}

pub(crate) fn encode_all_query_results<'tcx>(
    tcx: TyCtxt<'tcx>,
    encoder: &mut CacheEncoder<'_, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) {
    for_each_query_vtable!(CACHE_ON_DISK, tcx, |query| {
        encode_query_results(tcx, query, encoder, query_result_index)
    });
}

fn encode_query_results<'a, 'tcx, C, V>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    encoder: &mut CacheEncoder<'a, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) where
    C: QueryCache<Value = Erased<V>>,
    V: Erasable + Encodable<CacheEncoder<'a, 'tcx>>,
{
    let _timer = tcx.prof.generic_activity_with_arg("encode_query_results_for", query.name);

    assert!(all_inactive(&query.state));
    query.cache.for_each(&mut |key, value, dep_node| {
        if (query.will_cache_on_disk_for_key_fn)(tcx, *key) {
            let dep_node = SerializedDepNodeIndex::new(dep_node.index());

            // Record position of the cache entry.
            query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.position())));

            // Encode the type check tables with the `SerializedDepNodeIndex`
            // as tag.
            encoder.encode_tagged(dep_node, &erase::restore_val::<V>(*value));
        }
    });
}

pub(crate) fn query_key_hash_verify_all<'tcx>(tcx: TyCtxt<'tcx>) {
    if tcx.sess.opts.unstable_opts.incremental_verify_ich || cfg!(debug_assertions) {
        tcx.sess.time("query_key_hash_verify_all", || {
            for_each_query_vtable!(ALL, tcx, |query| {
                query_key_hash_verify(query, tcx);
            });
        });
    }
}

fn query_key_hash_verify<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
) {
    let _timer = tcx.prof.generic_activity_with_arg("query_key_hash_verify_for", query.name);

    let cache = &query.cache;
    let mut map = UnordMap::with_capacity(cache.len());
    cache.for_each(&mut |key, _, _| {
        let node = DepNode::construct(tcx, query.dep_kind, key);
        if let Some(other_key) = map.insert(node, *key) {
            bug!(
                "query key:\n\
                `{:?}`\n\
                and key:\n\
                `{:?}`\n\
                mapped to the same dep node:\n\
                {:?}",
                key,
                other_key,
                node
            );
        }
    });
}

/// Implementation of [`DepKindVTable::promote_from_disk_fn`] for queries.
pub(crate) fn promote_from_disk_inner<'tcx, Q: GetQueryVTable<'tcx>>(
    tcx: TyCtxt<'tcx>,
    dep_node: DepNode,
) {
    let query = Q::query_vtable(tcx);
    debug_assert!(tcx.dep_graph.is_green(&dep_node));

    let key = <Q::Cache as QueryCache>::Key::try_recover_key(tcx, &dep_node).unwrap_or_else(|| {
        panic!(
            "Failed to recover key for {dep_node:?} with key fingerprint {}",
            dep_node.key_fingerprint
        )
    });

    // If the recovered key isn't eligible for cache-on-disk, then there's no
    // value on disk to promote.
    if !(query.will_cache_on_disk_for_key_fn)(tcx, key) {
        return;
    }

    match query.cache.lookup(&key) {
        // If the value is already in memory, then promotion isn't needed.
        Some(_) => {}

        // "Execute" the query to load its disk-cached value into memory.
        //
        // We know that the key is cache-on-disk and its node is green,
        // so there _must_ be a value on disk to load.
        //
        // FIXME(Zalathar): Is there a reasonable way to skip more of the
        // query bookkeeping when doing this?
        None => {
            (query.execute_query_fn)(tcx, DUMMY_SP, key, QueryMode::Get);
        }
    }
}

pub(crate) fn loadable_from_disk<'tcx>(tcx: TyCtxt<'tcx>, id: SerializedDepNodeIndex) -> bool {
    if let Some(cache) = tcx.query_system.on_disk_cache.as_ref() {
        cache.loadable_from_disk(id)
    } else {
        false
    }
}

pub(crate) fn try_load_from_disk<'tcx, V>(
    tcx: TyCtxt<'tcx>,
    prev_index: SerializedDepNodeIndex,
    index: DepNodeIndex,
) -> Option<V>
where
    V: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
{
    let on_disk_cache = tcx.query_system.on_disk_cache.as_ref()?;

    let prof_timer = tcx.prof.incr_cache_loading();

    // The call to `with_query_deserialization` enforces that no new `DepNodes`
    // are created during deserialization. See the docs of that method for more
    // details.
    let value = tcx
        .dep_graph
        .with_query_deserialization(|| on_disk_cache.try_load_query_result(tcx, prev_index));

    prof_timer.finish_with_query_invocation_id(index.into());

    value
}

/// Implementation of [`DepKindVTable::force_from_dep_node_fn`] for queries.
pub(crate) fn force_from_dep_node_inner<'tcx, Q: GetQueryVTable<'tcx>>(
    tcx: TyCtxt<'tcx>,
    dep_node: DepNode,
    // Needed by the vtable function signature, but not used when forcing queries.
    _prev_index: SerializedDepNodeIndex,
) -> bool {
    let query = Q::query_vtable(tcx);

    // We must avoid ever having to call `force_from_dep_node()` for a
    // `DepNode::codegen_unit`:
    // Since we cannot reconstruct the query key of a `DepNode::codegen_unit`, we
    // would always end up having to evaluate the first caller of the
    // `codegen_unit` query that *is* reconstructible. This might very well be
    // the `compile_codegen_unit` query, thus re-codegenning the whole CGU just
    // to re-trigger calling the `codegen_unit` query with the right key. At
    // that point we would already have re-done all the work we are trying to
    // avoid doing in the first place.
    // The solution is simple: Just explicitly call the `codegen_unit` query for
    // each CGU, right after partitioning. This way `try_mark_green` will always
    // hit the cache instead of having to go through `force_from_dep_node`.
    // This assertion makes sure, we actually keep applying the solution above.
    debug_assert!(
        dep_node.kind != DepKind::codegen_unit,
        "calling force_from_dep_node() on dep_kinds::codegen_unit"
    );

    if let Some(key) = <Q::Cache as QueryCache>::Key::try_recover_key(tcx, &dep_node) {
        force_query(query, tcx, key, dep_node);
        true
    } else {
        false
    }
}
