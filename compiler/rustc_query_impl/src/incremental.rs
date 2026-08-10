use rustc_data_structures::fingerprint::{Fingerprint, PackedFingerprint};
use rustc_data_structures::unord::UnordMap;
use rustc_middle::bug;
#[expect(unused_imports, reason = "used by doc comments")]
use rustc_middle::dep_graph::DepKindVTable;
use rustc_middle::dep_graph::{
    DepGraphData, DepNode, DepNodeIndex, DepNodeKey, SerializedDepNodeIndex,
};
use rustc_middle::query::erase::{Erasable, Erased};
use rustc_middle::query::on_disk_cache::{CacheDecoder, CacheEncoder};
use rustc_middle::query::{QueryCache, QueryState, QueryVTable, erase};
use rustc_middle::ty::TyCtxt;
use rustc_middle::verify_ich::incremental_verify_ich;
use rustc_serialize::{Decodable, Encodable};

use crate::query_vtables::for_each_query_vtable;

fn all_inactive<'tcx, K>(state: &QueryState<'tcx, K>) -> bool {
    state.active.lock_shards().all(|shard| shard.is_empty())
}

pub(crate) fn encode_query_values<'tcx>(tcx: TyCtxt<'tcx>, encoder: &mut CacheEncoder<'_, 'tcx>) {
    for_each_query_vtable!(CACHE_ON_DISK, tcx, |query| {
        encode_query_values_inner(tcx, query, encoder)
    });
}

fn encode_query_values_inner<'a, 'tcx, C, V>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    encoder: &mut CacheEncoder<'a, 'tcx>,
) where
    C: QueryCache<Value = Erased<V>>,
    V: Erasable + Encodable<CacheEncoder<'a, 'tcx>>,
{
    let _timer = tcx.prof.generic_activity_with_arg("encode_query_results_for", query.name);

    assert!(all_inactive(&query.state));
    query.cache.for_each(&mut |key, value, dep_node| {
        if query.will_cache_on_disk_for_key(*key) {
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

fn verify_query_key_hashes_inner<'tcx, C: QueryCache>(
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

/// Whether a value loaded from the on-disk cache should have its fingerprint
/// verified with `incremental_verify_ich`. If `-Zincremental-verify-ich` is
/// specified, re-hash results from the cache and make sure that they have the
/// expected fingerprint.
///
/// If not, we still verify a subset: re-hashing is too expensive to do for
/// every value. The subset rotates with the session count, covering the whole
/// cache every 32 sessions, and is deterministic so that a verification
/// failure reproduces on retry.
///
/// `to_smaller_hash` mixes both fingerprint halves because neither half is
/// evenly distributed on its own (`DefPathHash` keys share the
/// `StableCrateId`, `HirId` keys contain a sequential id).
pub(crate) fn should_verify_loaded_value(
    tcx: TyCtxt<'_>,
    dep_graph_data: &DepGraphData,
    key_fingerprint: PackedFingerprint,
) -> bool {
    let hash = Fingerprint::from(key_fingerprint).to_smaller_hash().as_u64();
    hash % 32 == dep_graph_data.session_count() % 32
        || tcx.sess.opts.unstable_opts.incremental_verify_ich
}

/// Inner implementation of [`DepKindVTable::promote_from_disk_fn`] for queries.
pub(crate) fn promote_from_disk_inner<'tcx, C: QueryCache>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    dep_node: DepNode,
    prev_index: SerializedDepNodeIndex,
    dep_node_index: DepNodeIndex,
) {
    debug_assert!(tcx.dep_graph.is_green(&dep_node));

    let key = C::Key::try_recover_key(tcx, &dep_node).unwrap_or_else(|| {
        panic!(
            "Failed to recover key for {dep_node:?} with key fingerprint {}",
            dep_node.key_fingerprint
        )
    });

    // If the recovered key isn't eligible for cache-on-disk, then there's no
    // value on disk to promote.
    if !query.will_cache_on_disk_for_key(key) {
        return;
    }

    // If the value is already in memory, then promotion isn't needed.
    if query.cache.lookup(&key).is_some() {
        return;
    }

    // Load the disk-cached value into memory.
    let dep_graph_data =
        tcx.dep_graph.data().expect("should always be present in incremental mode");

    let prof_timer = tcx.prof.incr_cache_loading();
    let value = (query.try_load_from_disk_fn)(tcx, prev_index);
    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    let Some(value) = value else {
        // The key is cache-on-disk and its node is green, so a value must be on disk.
        bug!("failed to load disk-cached value for green node {dep_node:?}");
    };

    // Verify the fingerprints of the same subset of loaded values as
    // `load_from_disk_or_invoke_provider_green` does.
    if should_verify_loaded_value(tcx, dep_graph_data, dep_node.key_fingerprint) {
        incremental_verify_ich(
            tcx,
            dep_graph_data,
            &value,
            prev_index,
            query.hash_value_fn,
            query.format_value,
        );
    }

    query.cache.complete(key, value, dep_node_index);
}

pub(crate) fn try_load_from_disk<'tcx, V>(
    tcx: TyCtxt<'tcx>,
    prev_index: SerializedDepNodeIndex,
) -> Option<V>
where
    V: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
{
    let on_disk_cache = tcx.query_system.on_disk_cache.as_ref()?;

    // The call to `with_query_deserialization` enforces that no new `DepNodes`
    // are created during deserialization. See the docs of that method for more
    // details.
    tcx.dep_graph.with_query_deserialization(|| on_disk_cache.try_load_query_value(tcx, prev_index))
}
