//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

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
use rustc_middle::query::{
    QueryCache, QueryJobId, QueryKey, QueryStackDeferred, QueryStackFrame, QueryStackFrameExtra,
    erase,
};
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::print::with_reduced_queries;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::ty::{self, TyCtxt};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::def_id::LOCAL_CRATE;

use crate::error::{QueryOverflow, QueryOverflowNote};
use crate::execution::{all_inactive, force_query};
use crate::job::find_dep_kind_root;
use crate::{GetQueryVTable, collect_active_jobs_from_all_queries};

fn depth_limit_error<'tcx>(tcx: TyCtxt<'tcx>, job: QueryJobId) {
    let job_map =
        collect_active_jobs_from_all_queries(tcx, true).expect("failed to collect active queries");
    let (info, depth) = find_dep_kind_root(job, job_map);

    let suggested_limit = match tcx.recursion_limit() {
        Limit(0) => Limit(2),
        limit => limit * 2,
    };

    tcx.sess.dcx().emit_fatal(QueryOverflow {
        span: info.job.span,
        note: QueryOverflowNote { desc: info.frame.info.extract().description, depth },
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
pub(crate) fn current_query_job<'tcx>(tcx: TyCtxt<'tcx>) -> Option<QueryJobId> {
    tls::with_related_context(tcx, |icx| icx.query)
}

/// Executes a job by changing the `ImplicitCtxt` to point to the
/// new query job while it executes.
#[inline(always)]
pub(crate) fn start_query<'tcx, R>(
    tcx: TyCtxt<'tcx>,
    token: QueryJobId,
    depth_limit: bool,
    compute: impl FnOnce() -> R,
) -> R {
    // The `TyCtxt` stored in TLS has the same global interner lifetime
    // as `self`, so we use `with_related_context` to relate the 'tcx lifetimes
    // when accessing the `ImplicitCtxt`.
    tls::with_related_context(tcx, move |current_icx| {
        if depth_limit && !tcx.recursion_limit().value_within_limit(current_icx.query_depth) {
            depth_limit_error(tcx, token);
        }

        // Update the `ImplicitCtxt` to point to our new query job.
        let new_icx = ImplicitCtxt {
            tcx,
            query: Some(token),
            query_depth: current_icx.query_depth + depth_limit as usize,
            task_deps: current_icx.task_deps,
        };

        // Use the `ImplicitCtxt` while we execute the query.
        tls::enter_context(&new_icx, compute)
    })
}

pub(super) fn try_mark_green<'tcx>(tcx: TyCtxt<'tcx>, dep_node: &DepNode) -> bool {
    tcx.dep_graph.try_mark_green(tcx, dep_node).is_some()
}

macro_rules! cycle_error_handling {
    ([]) => {{
        rustc_middle::query::CycleErrorHandling::Error
    }};
    ([(cycle_fatal) $($rest:tt)*]) => {{
        rustc_middle::query::CycleErrorHandling::Fatal
    }};
    ([(cycle_stash) $($rest:tt)*]) => {{
        rustc_middle::query::CycleErrorHandling::Stash
    }};
    ([(cycle_delay_bug) $($rest:tt)*]) => {{
        rustc_middle::query::CycleErrorHandling::DelayBug
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        cycle_error_handling!([$($modifiers)*])
    };
}

macro_rules! is_anon {
    ([]) => {{
        false
    }};
    ([(anon) $($rest:tt)*]) => {{
        true
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        is_anon!([$($modifiers)*])
    };
}

macro_rules! is_eval_always {
    ([]) => {{
        false
    }};
    ([(eval_always) $($rest:tt)*]) => {{
        true
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        is_eval_always!([$($modifiers)*])
    };
}

macro_rules! is_depth_limit {
    ([]) => {{
        false
    }};
    ([(depth_limit) $($rest:tt)*]) => {{
        true
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        is_depth_limit!([$($modifiers)*])
    };
}

macro_rules! is_feedable {
    ([]) => {{
        false
    }};
    ([(feedable) $($rest:tt)*]) => {{
        true
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        is_feedable!([$($modifiers)*])
    };
}

/// Expands to `$yes` if the `no_hash` modifier is present, or `$no` otherwise.
macro_rules! if_no_hash {
    ([] $yes:tt $no:tt) => { $no };
    ([(no_hash) $($modifiers:tt)*] $yes:tt $no:tt) => { $yes };
    ([$other:tt $($modifiers:tt)*] $yes:tt $no:tt) => {
        if_no_hash!([$($modifiers)*] $yes $no)
    }
}

macro_rules! call_provider {
    ([][$tcx:expr, $name:ident, $key:expr]) => {{
        ($tcx.query_system.local_providers.$name)($tcx, $key)
    }};
    ([(separate_provide_extern) $($rest:tt)*][$tcx:expr, $name:ident, $key:expr]) => {{
        if let Some(key) = $key.as_local_key() {
            ($tcx.query_system.local_providers.$name)($tcx, key)
        } else {
            ($tcx.query_system.extern_providers.$name)($tcx, $key)
        }
    }};
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        call_provider!([$($modifiers)*][$($args)*])
    };
}

/// Expands to one of two token trees, depending on whether the current query
/// has the `cache_on_disk_if` modifier.
macro_rules! if_cache_on_disk {
    ([] $yes:tt $no:tt) => {
        $no
    };
    // The `cache_on_disk_if` modifier generates a synthetic `(cache_on_disk)`,
    // modifier, for use by this macro and similar macros.
    ([(cache_on_disk) $($rest:tt)*] $yes:tt $no:tt) => {
        $yes
    };
    ([$other:tt $($modifiers:tt)*] $yes:tt $no:tt) => {
        if_cache_on_disk!([$($modifiers)*] $yes $no)
    };
}

/// Conditionally expands to some token trees, if the current query has the
/// `cache_on_disk_if` modifier.
macro_rules! item_if_cache_on_disk {
    ([] $($item:tt)*) => {};
    ([(cache_on_disk) $($rest:tt)*] $($item:tt)*) => {
        $($item)*
    };
    ([$other:tt $($modifiers:tt)*] $($item:tt)*) => {
        item_if_cache_on_disk! { [$($modifiers)*] $($item)* }
    };
}

/// The deferred part of a deferred query stack frame.
fn mk_query_stack_frame_extra<'tcx, Cache>(
    (tcx, vtable, key): (TyCtxt<'tcx>, &'tcx QueryVTable<'tcx, Cache>, Cache::Key),
) -> QueryStackFrameExtra
where
    Cache: QueryCache,
    Cache::Key: QueryKey,
{
    let def_id = key.key_as_def_id();

    // If reduced queries are requested, we may be printing a query stack due
    // to a panic. Avoid using `default_span` and `def_kind` in that case.
    let reduce_queries = with_reduced_queries();

    // Avoid calling queries while formatting the description
    let description = ty::print::with_no_queries!((vtable.description_fn)(tcx, key));
    let description = if tcx.sess.verbose_internals() {
        format!("{description} [{name:?}]", name = vtable.name)
    } else {
        description
    };
    let span = if vtable.dep_kind == DepKind::def_span || reduce_queries {
        // The `def_span` query is used to calculate `default_span`,
        // so exit to avoid infinite recursion.
        None
    } else {
        Some(key.default_span(tcx))
    };

    let def_kind = if vtable.dep_kind == DepKind::def_kind || reduce_queries {
        // Try to avoid infinite recursion.
        None
    } else {
        def_id.and_then(|def_id| def_id.as_local()).map(|def_id| tcx.def_kind(def_id))
    };
    QueryStackFrameExtra::new(description, span, def_kind)
}

pub(crate) fn create_deferred_query_stack_frame<'tcx, C>(
    tcx: TyCtxt<'tcx>,
    vtable: &'tcx QueryVTable<'tcx, C>,
    key: C::Key,
) -> QueryStackFrame<QueryStackDeferred<'tcx>>
where
    C: QueryCache<Key: QueryKey + DynSend + DynSync>,
    QueryVTable<'tcx, C>: DynSync,
{
    let kind = vtable.dep_kind;

    let def_id: Option<DefId> = key.key_as_def_id();
    let def_id_for_ty_in_cycle: Option<DefId> = key.def_id_for_ty_in_cycle();

    let info = QueryStackDeferred::new((tcx, vtable, key), mk_query_stack_frame_extra);
    QueryStackFrame::new(info, kind, def_id, def_id_for_ty_in_cycle)
}

pub(crate) fn encode_query_results<'a, 'tcx, C, V>(
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
    query.cache.iter(&mut |key, value, dep_node| {
        if query.will_cache_on_disk_for_key(tcx, key) {
            let dep_node = SerializedDepNodeIndex::new(dep_node.index());

            // Record position of the cache entry.
            query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.position())));

            // Encode the type check tables with the `SerializedDepNodeIndex`
            // as tag.
            encoder.encode_tagged(dep_node, &erase::restore_val::<V>(*value));
        }
    });
}

pub(crate) fn query_key_hash_verify<'tcx, C: QueryCache>(
    query: &'tcx QueryVTable<'tcx, C>,
    tcx: TyCtxt<'tcx>,
) {
    let _timer = tcx.prof.generic_activity_with_arg("query_key_hash_verify_for", query.name);

    let cache = &query.cache;
    let mut map = UnordMap::with_capacity(cache.len());
    cache.iter(&mut |key, _, _| {
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
    if query.will_cache_on_disk_for_key(tcx, &key) {
        // Call `tcx.$query(key)` for its side-effect of loading the disk-cached
        // value into memory.
        (query.call_query_method_fn)(tcx, key);
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

// Note: `$K` and `$V` are unused but present so this can be called by `rustc_with_all_queries`.
macro_rules! define_queries {
    (
        $(
            $(#[$attr:meta])*
            [$($modifiers:tt)*] fn $name:ident($K:ty) -> $V:ty,
        )*
    ) => {

        pub(crate) mod query_impl { $(pub(crate) mod $name {
            use super::super::*;
            use ::rustc_middle::query::erase::{self, Erased};

            // It seems to be important that every query has its own monomorphic
            // copy of `execute_query_incr` and `execute_query_non_incr`.
            // Trying to inline these wrapper functions into their generic
            // "inner" helpers tends to break `tests/run-make/short-ice`.

            pub(crate) mod execute_query_incr {
                use super::*;

                // Adding `__rust_end_short_backtrace` marker to backtraces so that we emit the frames
                // when `RUST_BACKTRACE=1`, add a new mod with `$name` here is to allow duplicate naming
                #[inline(never)]
                pub(crate) fn __rust_end_short_backtrace<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    span: Span,
                    key: queries::$name::Key<'tcx>,
                    mode: QueryMode,
                ) -> Option<Erased<queries::$name::Value<'tcx>>> {
                    #[cfg(debug_assertions)]
                    let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();
                    execution::execute_query_incr_inner(
                        &tcx.query_system.query_vtables.$name,
                        tcx,
                        span,
                        key,
                        mode
                    )
                }
            }

            pub(crate) mod execute_query_non_incr {
                use super::*;

                #[inline(never)]
                pub(crate) fn __rust_end_short_backtrace<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    span: Span,
                    key: queries::$name::Key<'tcx>,
                    __mode: QueryMode,
                ) -> Option<Erased<queries::$name::Value<'tcx>>> {
                    Some(execution::execute_query_non_incr_inner(
                        &tcx.query_system.query_vtables.$name,
                        tcx,
                        span,
                        key,
                    ))
                }
            }

            /// Defines an `invoke_provider` function that calls the query's provider,
            /// to be used as a function pointer in the query's vtable.
            ///
            /// To mark a short-backtrace boundary, the function's actual name
            /// (after demangling) must be `__rust_begin_short_backtrace`.
            mod invoke_provider_fn {
                use super::*;
                use ::rustc_middle::queries::$name::{Key, Value, provided_to_erased};

                #[inline(never)]
                pub(crate) fn __rust_begin_short_backtrace<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    key: Key<'tcx>,
                ) -> Erased<Value<'tcx>> {
                    #[cfg(debug_assertions)]
                    let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();

                    // Call the actual provider function for this query.
                    let provided_value = call_provider!([$($modifiers)*][tcx, $name, key]);

                    rustc_middle::ty::print::with_reduced_queries!({
                        tracing::trace!(?provided_value);
                    });

                    // Erase the returned value, because `QueryVTable` uses erased values.
                    // For queries with `arena_cache`, this also arena-allocates the value.
                    provided_to_erased(tcx, provided_value)
                }
            }

            pub(crate) fn make_query_vtable<'tcx>(incremental: bool)
                -> QueryVTable<'tcx, queries::$name::Cache<'tcx>>
            {
                QueryVTable {
                    name: stringify!($name),
                    anon: is_anon!([$($modifiers)*]),
                    eval_always: is_eval_always!([$($modifiers)*]),
                    depth_limit: is_depth_limit!([$($modifiers)*]),
                    feedable: is_feedable!([$($modifiers)*]),
                    dep_kind: dep_graph::DepKind::$name,
                    cycle_error_handling: cycle_error_handling!([$($modifiers)*]),
                    state: Default::default(),
                    cache: Default::default(),
                    will_cache_on_disk_for_key_fn: if_cache_on_disk!([$($modifiers)*] {
                        Some(::rustc_middle::queries::_cache_on_disk_if_fns::$name)
                    } {
                        None
                    }),
                    call_query_method_fn: |tcx, key| {
                        // Call the query method for its side-effect of loading a value
                        // from disk-cache; the caller doesn't need the value.
                        let _ = tcx.$name(key);
                    },
                    invoke_provider_fn: self::invoke_provider_fn::__rust_begin_short_backtrace,
                    try_load_from_disk_fn: if_cache_on_disk!([$($modifiers)*] {
                        Some(|tcx, key, prev_index, index| {
                            // Check the `cache_on_disk_if` condition for this key.
                            if !::rustc_middle::queries::_cache_on_disk_if_fns::$name(tcx, key) {
                                return None;
                            }

                            let value: queries::$name::ProvidedValue<'tcx> =
                                $crate::plumbing::try_load_from_disk(tcx, prev_index, index)?;

                            // Arena-alloc the value if appropriate, and erase it.
                            Some(queries::$name::provided_to_erased(tcx, value))
                        })
                    } {
                        None
                    }),
                    is_loadable_from_disk_fn: if_cache_on_disk!([$($modifiers)*] {
                        Some(|tcx, key, index| -> bool {
                            ::rustc_middle::queries::_cache_on_disk_if_fns::$name(tcx, key) &&
                                $crate::plumbing::loadable_from_disk(tcx, index)
                        })
                    } {
                        None
                    }),
                    value_from_cycle_error: |tcx, cycle, guar| {
                        let result: queries::$name::Value<'tcx> = Value::from_cycle_error(tcx, cycle, guar);
                        erase::erase_val(result)
                    },
                    hash_value_fn: if_no_hash!(
                        [$($modifiers)*]
                        None
                        {
                            Some(|hcx, erased_value: &erase::Erased<queries::$name::Value<'tcx>>| {
                                let value = erase::restore_val(*erased_value);
                                rustc_middle::dep_graph::hash_result(hcx, &value)
                            })
                        }
                    ),
                    format_value: |value| format!("{:?}", erase::restore_val::<queries::$name::Value<'tcx>>(*value)),
                    description_fn: $crate::queries::_description_fns::$name,
                    execute_query_fn: if incremental {
                        query_impl::$name::execute_query_incr::__rust_end_short_backtrace
                    } else {
                        query_impl::$name::execute_query_non_incr::__rust_end_short_backtrace
                    },
                }
            }

            /// Marker type that implements [`GetQueryVTable`] for this query.
            pub(crate) enum VTableGetter {}

            impl<'tcx> GetQueryVTable<'tcx> for VTableGetter {
                type Cache = rustc_middle::queries::$name::Cache<'tcx>;

                #[inline(always)]
                fn query_vtable(tcx: TyCtxt<'tcx>) -> &'tcx QueryVTable<'tcx, Self::Cache> {
                    &tcx.query_system.query_vtables.$name
                }
            }
        })*}

        pub fn make_query_vtables<'tcx>(incremental: bool) -> queries::QueryVTables<'tcx> {
            queries::QueryVTables {
                $(
                    $name: query_impl::$name::make_query_vtable(incremental),
                )*
            }
        }

        /// Returns a map of currently active query jobs, collected from all queries.
        ///
        /// If `require_complete` is `true`, this function locks all shards of the
        /// query results to produce a complete map, which always returns `Ok`.
        /// Otherwise, it may return an incomplete map as an error if any shard
        /// lock cannot be acquired.
        ///
        /// Prefer passing `false` to `require_complete` to avoid potential deadlocks,
        /// especially when called from within a deadlock handler, unless a
        /// complete map is needed and no deadlock is possible at this call site.
        pub fn collect_active_jobs_from_all_queries<'tcx>(
            tcx: TyCtxt<'tcx>,
            require_complete: bool,
        ) -> Result<QueryJobMap<'tcx>, QueryJobMap<'tcx>> {
            let mut job_map_out = QueryJobMap::default();
            let mut complete = true;

            $(
                let res = crate::execution::gather_active_jobs(
                    &tcx.query_system.query_vtables.$name,
                    tcx,
                    require_complete,
                    &mut job_map_out,
                );
                if res.is_none() {
                    complete = false;
                }
            )*

            if complete { Ok(job_map_out) } else { Err(job_map_out) }
        }

        /// All self-profiling events generated by the query engine use
        /// virtual `StringId`s for their `event_id`. This method makes all
        /// those virtual `StringId`s point to actual strings.
        ///
        /// If we are recording only summary data, the ids will point to
        /// just the query names. If we are recording query keys too, we
        /// allocate the corresponding strings here.
        pub fn alloc_self_profile_query_strings(tcx: TyCtxt<'_>) {
            if !tcx.prof.enabled() {
                return;
            }

            let _prof_timer = tcx.sess.prof.generic_activity("self_profile_alloc_query_strings");

            let mut string_cache = QueryKeyStringCache::new();

            $(
                $crate::profiling_support::alloc_self_profile_query_strings_for_query_cache(
                    tcx,
                    stringify!($name),
                    &tcx.query_system.query_vtables.$name.cache,
                    &mut string_cache,
                );
            )*

            tcx.sess.prof.store_query_cache_hits();
        }

        fn encode_all_query_results<'tcx>(
            tcx: TyCtxt<'tcx>,
            encoder: &mut CacheEncoder<'_, 'tcx>,
            query_result_index: &mut EncodedDepNodeIndex,
        ) {
            $(
                item_if_cache_on_disk! {
                    [$($modifiers)*]
                    $crate::plumbing::encode_query_results(
                        tcx,
                        &tcx.query_system.query_vtables.$name,
                        encoder,
                        query_result_index,
                    )
                }
            )*
        }

        pub fn query_key_hash_verify_all<'tcx>(tcx: TyCtxt<'tcx>) {
            if tcx.sess.opts.unstable_opts.incremental_verify_ich || cfg!(debug_assertions) {
                tcx.sess.time("query_key_hash_verify_all", || {
                    $(
                        $crate::plumbing::query_key_hash_verify(
                            &tcx.query_system.query_vtables.$name,
                            tcx
                        );
                    )*
                })
            }
        }

        /// Declares a dep-kind vtable constructor for each query.
        mod _dep_kind_vtable_ctors_for_queries {
            use ::rustc_middle::dep_graph::DepKindVTable;
            use $crate::dep_kind_vtables::make_dep_kind_vtable_for_query;

            $(
                /// `DepKindVTable` constructor for this query.
                pub(crate) fn $name<'tcx>() -> DepKindVTable<'tcx> {
                    use $crate::query_impl::$name::VTableGetter;
                    make_dep_kind_vtable_for_query::<VTableGetter>(
                        is_anon!([$($modifiers)*]),
                        if_cache_on_disk!([$($modifiers)*] true false),
                        is_eval_always!([$($modifiers)*]),
                    )
                }
            )*
        }
    }
}
