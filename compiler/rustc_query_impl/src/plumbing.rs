//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use std::num::NonZero;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_data_structures::unord::UnordMap;
use rustc_hashes::Hash64;
use rustc_hir::def_id::DefId;
use rustc_hir::limit::Limit;
use rustc_index::Idx;
use rustc_middle::bug;
#[expect(unused_imports, reason = "used by doc comments")]
use rustc_middle::dep_graph::DepKindVTable;
use rustc_middle::dep_graph::{
    self, DepNode, DepNodeIndex, DepNodeKey, HasDepContext, SerializedDepNodeIndex, dep_kinds,
};
use rustc_middle::query::on_disk_cache::{
    AbsoluteBytePos, CacheDecoder, CacheEncoder, EncodedDepNodeIndex,
};
use rustc_middle::query::plumbing::QueryVTable;
use rustc_middle::query::{
    Key, QueryCache, QueryJobId, QueryStackDeferred, QueryStackFrame, QueryStackFrameExtra,
};
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::print::with_reduced_queries;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::ty::{self, TyCtxt};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::def_id::LOCAL_CRATE;

use crate::error::{QueryOverflow, QueryOverflowNote};
use crate::execution::{all_inactive, force_query};
use crate::job::{QueryJobMap, find_dep_kind_root};
use crate::{QueryDispatcherUnerased, QueryFlags, SemiDynamicQueryDispatcher};

#[derive(Copy, Clone)]
pub struct QueryCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

impl<'tcx> QueryCtxt<'tcx> {
    #[inline]
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        QueryCtxt { tcx }
    }

    fn depth_limit_error(self, job: QueryJobId) {
        let job_map = self
            .collect_active_jobs_from_all_queries(true)
            .expect("failed to collect active queries");
        let (info, depth) = find_dep_kind_root(job, job_map);

        let suggested_limit = match self.tcx.recursion_limit() {
            Limit(0) => Limit(2),
            limit => limit * 2,
        };

        self.tcx.sess.dcx().emit_fatal(QueryOverflow {
            span: info.job.span,
            note: QueryOverflowNote { desc: info.frame.info.extract().description, depth },
            suggested_limit,
            crate_name: self.tcx.crate_name(LOCAL_CRATE),
        });
    }

    #[inline]
    pub(crate) fn next_job_id(self) -> QueryJobId {
        QueryJobId(
            NonZero::new(
                self.tcx.query_system.jobs.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            )
            .unwrap(),
        )
    }

    #[inline]
    pub(crate) fn current_query_job(self) -> Option<QueryJobId> {
        tls::with_related_context(self.tcx, |icx| icx.query)
    }

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes.
    #[inline(always)]
    pub(crate) fn start_query<R>(
        self,
        token: QueryJobId,
        depth_limit: bool,
        compute: impl FnOnce() -> R,
    ) -> R {
        // The `TyCtxt` stored in TLS has the same global interner lifetime
        // as `self`, so we use `with_related_context` to relate the 'tcx lifetimes
        // when accessing the `ImplicitCtxt`.
        tls::with_related_context(self.tcx, move |current_icx| {
            if depth_limit
                && !self.tcx.recursion_limit().value_within_limit(current_icx.query_depth)
            {
                self.depth_limit_error(token);
            }

            // Update the `ImplicitCtxt` to point to our new query job.
            let new_icx = ImplicitCtxt {
                tcx: self.tcx,
                query: Some(token),
                query_depth: current_icx.query_depth + depth_limit as usize,
                task_deps: current_icx.task_deps,
            };

            // Use the `ImplicitCtxt` while we execute the query.
            tls::enter_context(&new_icx, compute)
        })
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
    pub fn collect_active_jobs_from_all_queries(
        self,
        require_complete: bool,
    ) -> Result<QueryJobMap<'tcx>, QueryJobMap<'tcx>> {
        let mut job_map_out = QueryJobMap::default();
        let mut complete = true;

        for gather_fn in crate::PER_QUERY_GATHER_ACTIVE_JOBS_FNS.iter() {
            if gather_fn(self.tcx, require_complete, &mut job_map_out).is_none() {
                complete = false;
            }
        }

        if complete { Ok(job_map_out) } else { Err(job_map_out) }
    }
}

impl<'tcx> HasDepContext<'tcx> for QueryCtxt<'tcx> {
    #[inline]
    fn dep_context(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

pub(super) fn try_mark_green<'tcx>(tcx: TyCtxt<'tcx>, dep_node: &dep_graph::DepNode) -> bool {
    tcx.dep_graph.try_mark_green(tcx, dep_node).is_some()
}

pub(super) fn encode_all_query_results<'tcx>(
    tcx: TyCtxt<'tcx>,
    encoder: &mut CacheEncoder<'_, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) {
    for encode in super::ENCODE_QUERY_RESULTS.iter().copied().flatten() {
        encode(tcx, encoder, query_result_index);
    }
}

pub fn query_key_hash_verify_all<'tcx>(tcx: TyCtxt<'tcx>) {
    if tcx.sess.opts.unstable_opts.incremental_verify_ich || cfg!(debug_assertions) {
        tcx.sess.time("query_key_hash_verify_all", || {
            for verify in super::QUERY_KEY_HASH_VERIFY.iter() {
                verify(tcx);
            }
        })
    }
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

macro_rules! depth_limit {
    ([]) => {{
        false
    }};
    ([(depth_limit) $($rest:tt)*]) => {{
        true
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        depth_limit!([$($modifiers)*])
    };
}

macro_rules! feedable {
    ([]) => {{
        false
    }};
    ([(feedable) $($rest:tt)*]) => {{
        true
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        feedable!([$($modifiers)*])
    };
}

macro_rules! hash_result {
    ([][$V:ty]) => {{
        Some(|hcx, result| {
            let result = rustc_middle::query::erase::restore_val::<$V>(*result);
            rustc_middle::dep_graph::hash_result(hcx, &result)
        })
    }};
    ([(no_hash) $($rest:tt)*][$V:ty]) => {{
        None
    }};
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        hash_result!([$($modifiers)*][$($args)*])
    };
}

macro_rules! call_provider {
    ([][$tcx:expr, $name:ident, $key:expr]) => {{
        ($tcx.query_system.fns.local_providers.$name)($tcx, $key)
    }};
    ([(separate_provide_extern) $($rest:tt)*][$tcx:expr, $name:ident, $key:expr]) => {{
        if let Some(key) = $key.as_local_key() {
            ($tcx.query_system.fns.local_providers.$name)($tcx, key)
        } else {
            ($tcx.query_system.fns.extern_providers.$name)($tcx, $key)
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
    Cache::Key: Key,
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
    let span = if vtable.dep_kind == dep_graph::dep_kinds::def_span || reduce_queries {
        // The `def_span` query is used to calculate `default_span`,
        // so exit to avoid infinite recursion.
        None
    } else {
        Some(key.default_span(tcx))
    };

    let def_kind = if vtable.dep_kind == dep_graph::dep_kinds::def_kind || reduce_queries {
        // Try to avoid infinite recursion.
        None
    } else {
        def_id.and_then(|def_id| def_id.as_local()).map(|def_id| tcx.def_kind(def_id))
    };
    QueryStackFrameExtra::new(description, span, def_kind)
}

pub(crate) fn create_deferred_query_stack_frame<'tcx, Cache>(
    tcx: TyCtxt<'tcx>,
    vtable: &'tcx QueryVTable<'tcx, Cache>,
    key: Cache::Key,
) -> QueryStackFrame<QueryStackDeferred<'tcx>>
where
    Cache: QueryCache,
    Cache::Key: Key + DynSend + DynSync,
{
    let kind = vtable.dep_kind;

    let hash = tcx.with_stable_hashing_context(|mut hcx| {
        let mut hasher = StableHasher::new();
        kind.as_usize().hash_stable(&mut hcx, &mut hasher);
        key.hash_stable(&mut hcx, &mut hasher);
        hasher.finish::<Hash64>()
    });

    let def_id: Option<DefId> = key.key_as_def_id();
    let def_id_for_ty_in_cycle: Option<DefId> = key.def_id_for_ty_in_cycle();

    let info = QueryStackDeferred::new((tcx, vtable, key), mk_query_stack_frame_extra);
    QueryStackFrame::new(info, kind, hash, def_id, def_id_for_ty_in_cycle)
}

pub(crate) fn encode_query_results<'a, 'tcx, Q, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
    encoder: &mut CacheEncoder<'a, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) where
    Q: QueryDispatcherUnerased<'tcx, C, FLAGS>,
    Q::UnerasedValue: Encodable<CacheEncoder<'a, 'tcx>>,
{
    let _timer = qcx.tcx.prof.generic_activity_with_arg("encode_query_results_for", query.name());

    assert!(all_inactive(query.query_state(qcx)));
    let cache = query.query_cache(qcx);
    cache.iter(&mut |key, value, dep_node| {
        if query.will_cache_on_disk_for_key(qcx.tcx, key) {
            let dep_node = SerializedDepNodeIndex::new(dep_node.index());

            // Record position of the cache entry.
            query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.position())));

            // Encode the type check tables with the `SerializedDepNodeIndex`
            // as tag.
            encoder.encode_tagged(dep_node, &Q::restore_val(*value));
        }
    });
}

pub(crate) fn query_key_hash_verify<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    qcx: QueryCtxt<'tcx>,
) {
    let _timer = qcx.tcx.prof.generic_activity_with_arg("query_key_hash_verify_for", query.name());

    let cache = query.query_cache(qcx);
    let mut map = UnordMap::with_capacity(cache.len());
    cache.iter(&mut |key, _, _| {
        let node = DepNode::construct(qcx.tcx, query.dep_kind(), key);
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

/// Implementation of [`DepKindVTable::try_load_from_on_disk_cache`] for queries.
pub(crate) fn try_load_from_on_disk_cache_inner<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    tcx: TyCtxt<'tcx>,
    dep_node: DepNode,
) {
    debug_assert!(tcx.dep_graph.is_green(&dep_node));

    let key = C::Key::recover(tcx, &dep_node).unwrap_or_else(|| {
        panic!("Failed to recover key for {:?} with hash {}", dep_node, dep_node.hash)
    });
    if query.will_cache_on_disk_for_key(tcx, &key) {
        // Call `tcx.$query(key)` for its side-effect of loading the disk-cached
        // value into memory.
        query.call_query_method(tcx, key);
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

/// Implementation of [`DepKindVTable::force_from_dep_node`] for queries.
pub(crate) fn force_from_dep_node_inner<'tcx, C: QueryCache, const FLAGS: QueryFlags>(
    query: SemiDynamicQueryDispatcher<'tcx, C, FLAGS>,
    tcx: TyCtxt<'tcx>,
    dep_node: DepNode,
) -> bool {
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
        dep_node.kind != dep_kinds::codegen_unit,
        "calling force_from_dep_node() on dep_kinds::codegen_unit"
    );

    if let Some(key) = C::Key::recover(tcx, &dep_node) {
        force_query(query, QueryCtxt::new(tcx), key, dep_node);
        true
    } else {
        false
    }
}

// NOTE: `$V` isn't used here, but we still need to match on it so it can be passed to other macros
// invoked by `rustc_with_all_queries`.
macro_rules! define_queries {
    (
        $(
            $(#[$attr:meta])*
            [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,
        )*
    ) => {

        pub(crate) mod query_impl { $(pub(crate) mod $name {
            use super::super::*;
            use std::marker::PhantomData;
            use ::rustc_middle::query::erase::{self, Erased};

            pub(crate) mod get_query_incr {
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
                    execution::get_query_incr(
                        QueryType::query_dispatcher(tcx),
                        QueryCtxt::new(tcx),
                        span,
                        key,
                        mode
                    )
                }
            }

            pub(crate) mod get_query_non_incr {
                use super::*;

                #[inline(never)]
                pub(crate) fn __rust_end_short_backtrace<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    span: Span,
                    key: queries::$name::Key<'tcx>,
                    __mode: QueryMode,
                ) -> Option<Erased<queries::$name::Value<'tcx>>> {
                    Some(execution::get_query_non_incr(
                        QueryType::query_dispatcher(tcx),
                        QueryCtxt::new(tcx),
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

            pub(crate) fn make_query_vtable<'tcx>()
                -> QueryVTable<'tcx, queries::$name::Storage<'tcx>>
            {
                QueryVTable {
                    name: stringify!($name),
                    eval_always: is_eval_always!([$($modifiers)*]),
                    dep_kind: dep_graph::dep_kinds::$name,
                    cycle_error_handling: cycle_error_handling!([$($modifiers)*]),
                    query_state: std::mem::offset_of!(QueryStates<'tcx>, $name),
                    query_cache: std::mem::offset_of!(QueryCaches<'tcx>, $name),
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
                    hash_result: hash_result!([$($modifiers)*][queries::$name::Value<'tcx>]),
                    format_value: |value| format!("{:?}", erase::restore_val::<queries::$name::Value<'tcx>>(*value)),
                    description_fn: $crate::queries::_description_fns::$name,
                }
            }

            #[derive(Copy, Clone, Default)]
            pub(crate) struct QueryType<'tcx> {
                data: PhantomData<&'tcx ()>
            }

            const FLAGS: QueryFlags = QueryFlags {
                is_anon: is_anon!([$($modifiers)*]),
                is_depth_limit: depth_limit!([$($modifiers)*]),
                is_feedable: feedable!([$($modifiers)*]),
            };

            impl<'tcx> QueryDispatcherUnerased<'tcx, queries::$name::Storage<'tcx>, FLAGS>
                for QueryType<'tcx>
            {
                type UnerasedValue = queries::$name::Value<'tcx>;

                const NAME: &'static &'static str = &stringify!($name);

                #[inline(always)]
                fn query_dispatcher(tcx: TyCtxt<'tcx>)
                    -> SemiDynamicQueryDispatcher<'tcx, queries::$name::Storage<'tcx>, FLAGS>
                {
                    SemiDynamicQueryDispatcher {
                        vtable: &tcx.query_system.query_vtables.$name,
                    }
                }

                #[inline(always)]
                fn restore_val(value: <queries::$name::Storage<'tcx> as QueryCache>::Value)
                    -> Self::UnerasedValue
                {
                    erase::restore_val::<queries::$name::Value<'tcx>>(value)
                }
            }

            /// Internal per-query plumbing for collecting the set of active jobs for this query.
            ///
            /// Should only be called through `PER_QUERY_GATHER_ACTIVE_JOBS_FNS`.
            pub(crate) fn gather_active_jobs<'tcx>(
                tcx: TyCtxt<'tcx>,
                require_complete: bool,
                job_map_out: &mut QueryJobMap<'tcx>,
            ) -> Option<()> {
                let make_frame = |tcx: TyCtxt<'tcx>, key| {
                    let vtable = &tcx.query_system.query_vtables.$name;
                    $crate::plumbing::create_deferred_query_stack_frame(tcx, vtable, key)
                };

                // Call `gather_active_jobs_inner` to do the actual work.
                let res = crate::execution::gather_active_jobs_inner(&tcx.query_system.states.$name,
                    tcx,
                    make_frame,
                    require_complete,
                    job_map_out,
                );

                // this can be called during unwinding, and the function has a `try_`-prefix, so
                // don't `unwrap()` here, just manually check for `None` and do best-effort error
                // reporting.
                if res.is_none() {
                    tracing::warn!(
                        "Failed to collect active jobs for query with name `{}`!",
                        stringify!($name)
                    );
                }
                res
            }

            pub(crate) fn alloc_self_profile_query_strings<'tcx>(
                tcx: TyCtxt<'tcx>,
                string_cache: &mut QueryKeyStringCache
            ) {
                $crate::profiling_support::alloc_self_profile_query_strings_for_query_cache(
                    tcx,
                    stringify!($name),
                    &tcx.query_system.caches.$name,
                    string_cache,
                )
            }

            item_if_cache_on_disk! { [$($modifiers)*]
                pub(crate) fn encode_query_results<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    encoder: &mut CacheEncoder<'_, 'tcx>,
                    query_result_index: &mut EncodedDepNodeIndex
                ) {
                    $crate::plumbing::encode_query_results::<
                        query_impl::$name::QueryType<'tcx>,
                        _,
                        _
                    > (
                        query_impl::$name::QueryType::query_dispatcher(tcx),
                        QueryCtxt::new(tcx),
                        encoder,
                        query_result_index,
                    )
                }
            }

            pub(crate) fn query_key_hash_verify<'tcx>(tcx: TyCtxt<'tcx>) {
                $crate::plumbing::query_key_hash_verify(
                    query_impl::$name::QueryType::query_dispatcher(tcx),
                    QueryCtxt::new(tcx),
                )
            }
        })*}

        pub(crate) fn engine(incremental: bool) -> QueryEngine {
            if incremental {
                QueryEngine {
                    $($name: query_impl::$name::get_query_incr::__rust_end_short_backtrace,)*
                }
            } else {
                QueryEngine {
                    $($name: query_impl::$name::get_query_non_incr::__rust_end_short_backtrace,)*
                }
            }
        }

        pub fn make_query_vtables<'tcx>() -> queries::PerQueryVTables<'tcx> {
            queries::PerQueryVTables {
                $(
                    $name: query_impl::$name::make_query_vtable(),
                )*
            }
        }

        // These arrays are used for iteration and can't be indexed by `DepKind`.

        /// Used by `collect_active_jobs_from_all_queries` to iterate over all
        /// queries, and gather the active jobs for each query.
        ///
        /// (We arbitrarily use the word "gather" when collecting the jobs for
        /// each individual query, so that we have distinct function names to
        /// grep for.)
        const PER_QUERY_GATHER_ACTIVE_JOBS_FNS: &[
            for<'tcx> fn(
                tcx: TyCtxt<'tcx>,
                require_complete: bool,
                job_map_out: &mut QueryJobMap<'tcx>,
            ) -> Option<()>
        ] = &[
            $( $crate::query_impl::$name::gather_active_jobs ),*
        ];

        const ALLOC_SELF_PROFILE_QUERY_STRINGS: &[
            for<'tcx> fn(TyCtxt<'tcx>, &mut QueryKeyStringCache)
        ] = &[$(query_impl::$name::alloc_self_profile_query_strings),*];

        const ENCODE_QUERY_RESULTS: &[
            Option<for<'tcx> fn(
                TyCtxt<'tcx>,
                &mut CacheEncoder<'_, 'tcx>,
                &mut EncodedDepNodeIndex)
            >
        ] = &[
            $(
                if_cache_on_disk!([$($modifiers)*] {
                    Some(query_impl::$name::encode_query_results)
                } {
                    None
                })
            ),*
        ];

        const QUERY_KEY_HASH_VERIFY: &[
            for<'tcx> fn(TyCtxt<'tcx>)
        ] = &[$(query_impl::$name::query_key_hash_verify),*];

        /// Declares a dep-kind vtable constructor for each query.
        mod _dep_kind_vtable_ctors_for_queries {
            use ::rustc_middle::dep_graph::DepKindVTable;
            use $crate::dep_kind_vtables::make_dep_kind_vtable_for_query;

            $(
                /// `DepKindVTable` constructor for this query.
                pub(crate) fn $name<'tcx>() -> DepKindVTable<'tcx> {
                    use $crate::query_impl::$name::QueryType;
                    make_dep_kind_vtable_for_query::<QueryType<'tcx>, _, _>(
                        is_eval_always!([$($modifiers)*]),
                    )
                }
            )*
        }
    }
}
