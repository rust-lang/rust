use rustc_middle::query::QueryCache;
use rustc_middle::query::plumbing::QueryVTable;
use rustc_middle::ty::TyCtxt;
use rustc_middle::{bug, dep_graph};

/// Called by the macro-generated impl of [`QueryVTable::feed_fn`].
pub(crate) fn feed_query_inner<'tcx, C: QueryCache>(
    tcx: TyCtxt<'tcx>,
    query: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
    value: C::Value,
) {
    let format_value = query.format_value;

    // Check whether the in-memory cache already has a value for this key.
    match rustc_middle::query::try_get_cached(tcx, &query.cache, &key) {
        Some(old) => {
            // The query already has a cached value for this key.
            // That's OK if both values are the same, i.e. they have the same hash,
            // so now we check their hashes.
            if let Some(hash_value_fn) = query.hash_value_fn {
                let (old_hash, value_hash) = tcx.with_stable_hashing_context(|ref mut hcx| {
                    (hash_value_fn(hcx, &old), hash_value_fn(hcx, &value))
                });
                if old_hash != value_hash {
                    // We have an inconsistency. This can happen if one of the two
                    // results is tainted by errors. In this case, delay a bug to
                    // ensure compilation is doomed, and keep the `old` value.
                    tcx.dcx().delayed_bug(format!(
                        "Trying to feed an already recorded value for query {query:?} key={key:?}:\n\
                        old value: {old}\nnew value: {value}",
                        old = format_value(&old),
                        value = format_value(&value),
                    ));
                }
            } else {
                // The query is `no_hash`, so we have no way to perform a sanity check.
                // If feeding the same value multiple times needs to be supported,
                // the query should not be marked `no_hash`.
                bug!(
                    "Trying to feed an already recorded value for query {query:?} key={key:?}:\n\
                    old value: {old}\nnew value: {value}",
                    old = format_value(&old),
                    value = format_value(&value),
                )
            }
        }
        None => {
            // There is no cached value for this key, so feed the query by
            // adding the provided value to the cache.
            let dep_node = dep_graph::DepNode::construct(tcx, query.dep_kind, &key);
            let dep_node_index = tcx.dep_graph.with_feed_task(tcx, query, dep_node, &value);
            query.cache.complete(key, value, dep_node_index);
        }
    }
}
