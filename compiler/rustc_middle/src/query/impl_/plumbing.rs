use std::num::NonZero;

use rustc_data_structures::unord::UnordMap;
use rustc_hir::limit::Limit;
#[expect(unused_imports, reason = "used by doc comments")]
use rustc_middle::dep_graph::DepKindVTable;
use rustc_middle::dep_graph::{DepNode, DepNodeKey, SerializedDepNodeIndex};
use rustc_middle::query::erase::{Erasable, Erased};
use rustc_middle::query::on_disk_cache::CacheEncoder;
use rustc_middle::query::{QueryCache, QueryHelper, QueryJobId, QueryMode, QueryVTable, erase};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_serialize::Encodable;
use rustc_span::DUMMY_SP;
use rustc_span::def_id::LOCAL_CRATE;

use crate::query::impl_::error::{QueryOverflow, QueryOverflowNote};
use crate::query::impl_::execution::all_inactive;
use crate::query::impl_::job::find_dep_kind_root;
use crate::query::impl_::query_impl::for_each_query_vtable;
use crate::query::impl_::{CollectActiveJobsKind, collect_active_query_jobs};

fn depth_limit_error<'tcx>(tcx: TyCtxt<'tcx>, job: QueryJobId) {
    let job_map = collect_active_query_jobs(tcx, CollectActiveJobsKind::Full);
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

pub(crate) fn encode_query_values<'tcx>(tcx: TyCtxt<'tcx>, encoder: &mut CacheEncoder<'_, 'tcx>) {
    for_each_query_vtable!(CACHE_ON_DISK, tcx, |query| {
        encode_query_values_inner(tcx, query, encoder)
    });
}

fn encode_query_values_inner<'a, 'tcx, C, V, H>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C, H>,
    encoder: &mut CacheEncoder<'a, 'tcx>,
) where
    C: QueryCache<Value = Erased<V>>,
    H: QueryHelper<'tcx, C::Key, C::Value>,
    V: Erasable + Encodable<CacheEncoder<'a, 'tcx>>,
{
    let _timer = tcx.prof.generic_activity_with_arg("encode_query_results_for", query.name);

    assert!(all_inactive(&query.state));
    query.cache.for_each(&mut |key, value, dep_node| {
        if H::will_cache_on_disk_for_key(*key) {
            encoder.encode_query_value::<V>(dep_node, &erase::restore_val::<V>(*value));
        }
    });
}

pub(crate) fn verify_query_key_hashes<'tcx>(tcx: TyCtxt<'tcx>) {
    if tcx.sess.opts.unstable_opts.incremental_verify_ich || cfg!(debug_assertions) {
        tcx.sess.time("verify_query_key_hashes", || {
            for_each_query_vtable!(ALL, tcx, |query| {
                verify_query_key_hashes_inner(query, tcx);
            });
        });
    }
}

fn verify_query_key_hashes_inner<'tcx, C: QueryCache, H: QueryHelper<'tcx, C::Key, C::Value>>(
    query: &'tcx QueryVTable<'tcx, C, H>,
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

/// Inner implementation of [`DepKindVTable::promote_from_disk_fn`] for queries.
pub(crate) fn promote_from_disk_inner<'tcx, C: QueryCache, H>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C, H>,
    dep_node: DepNode,
) where
    H: QueryHelper<'tcx, C::Key, C::Value>,
{
    debug_assert!(tcx.dep_graph.is_green(&dep_node));

    let key = C::Key::try_recover_key(tcx, &dep_node).unwrap_or_else(|| {
        panic!(
            "Failed to recover key for {dep_node:?} with key fingerprint {}",
            dep_node.key_fingerprint
        )
    });

    // If the recovered key isn't eligible for cache-on-disk, then there's no
    // value on disk to promote.
    if !H::will_cache_on_disk_for_key(key) {
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
            query.execute_query_fn(tcx, DUMMY_SP, key, QueryMode::Get);
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
