//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use std::hash::Hash;
use std::mem;
use std::num::NonZero;

use rustc_data_structures::hash_table::{Entry, HashTable};
use rustc_data_structures::jobserver::Proxy;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::{self, DynSend, DynSync};
use rustc_data_structures::unord::UnordMap;
use rustc_data_structures::{outline, sharded};
use rustc_errors::{Diag, FatalError, StashKey};
use rustc_hashes::Hash64;
use rustc_hir::def_id::DefId;
use rustc_hir::limit::Limit;
use rustc_index::Idx;
use rustc_middle::bug;
use rustc_middle::dep_graph::{
    self, DepContext, DepKindVTable, DepNode, DepNodeIndex, SerializedDepNodeIndex, dep_kinds,
};
use rustc_middle::query::Key;
use rustc_middle::query::on_disk_cache::{
    AbsoluteBytePos, CacheDecoder, CacheEncoder, EncodedDepNodeIndex,
};
use rustc_middle::query::plumbing::QueryVTable;
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::print::with_reduced_queries;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::ty::{self, TyCtxt};
use rustc_query_system::dep_graph::{DepGraphData, DepNodeKey, FingerprintStyle, HasDepContext};
use rustc_query_system::ich::StableHashingContext;
use rustc_query_system::query::{
    ActiveKeyStatus, CycleError, CycleErrorHandling, QueryCache, QueryContext, QueryDispatcher,
    QueryJob, QueryJobId, QueryJobInfo, QueryLatch, QueryMap, QueryMode, QuerySideEffect,
    QueryStackDeferred, QueryStackFrame, QueryStackFrameExtra, QueryState, incremental_verify_ich,
    report_cycle,
};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::{DUMMY_SP, Span};

use crate::QueryDispatcherUnerased;
use crate::error::{QueryOverflow, QueryOverflowNote};

/// Implements [`QueryContext`] for use by [`rustc_query_system`], since that
/// crate does not have direct access to [`TyCtxt`].
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
        let query_map = self
            .collect_active_jobs_from_all_queries(true)
            .expect("failed to collect active queries");
        let (info, depth) = job.find_dep_kind_root(query_map);

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
}

impl<'tcx> HasDepContext for QueryCtxt<'tcx> {
    type Deps = rustc_middle::dep_graph::DepsType;
    type DepContext = TyCtxt<'tcx>;

    #[inline]
    fn dep_context(&self) -> &Self::DepContext {
        &self.tcx
    }
}

impl<'tcx> QueryContext<'tcx> for QueryCtxt<'tcx> {
    #[inline]
    fn jobserver_proxy(&self) -> &Proxy {
        &self.tcx.jobserver_proxy
    }

    #[inline]
    fn next_job_id(self) -> QueryJobId {
        QueryJobId(
            NonZero::new(
                self.tcx.query_system.jobs.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            )
            .unwrap(),
        )
    }

    #[inline]
    fn current_query_job(self) -> Option<QueryJobId> {
        tls::with_related_context(self.tcx, |icx| icx.query)
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
    fn collect_active_jobs_from_all_queries(
        self,
        require_complete: bool,
    ) -> Result<QueryMap<'tcx>, QueryMap<'tcx>> {
        let mut jobs = QueryMap::default();
        let mut complete = true;

        for gather_fn in crate::PER_QUERY_GATHER_ACTIVE_JOBS_FNS.iter() {
            if gather_fn(self.tcx, &mut jobs, require_complete).is_none() {
                complete = false;
            }
        }

        if complete { Ok(jobs) } else { Err(jobs) }
    }

    // Interactions with on_disk_cache
    fn load_side_effect(
        self,
        prev_dep_node_index: SerializedDepNodeIndex,
    ) -> Option<QuerySideEffect> {
        self.tcx
            .query_system
            .on_disk_cache
            .as_ref()
            .and_then(|c| c.load_side_effect(self.tcx, prev_dep_node_index))
    }

    #[inline(never)]
    #[cold]
    fn store_side_effect(self, dep_node_index: DepNodeIndex, side_effect: QuerySideEffect) {
        if let Some(c) = self.tcx.query_system.on_disk_cache.as_ref() {
            c.store_side_effect(dep_node_index, side_effect)
        }
    }

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes.
    #[inline(always)]
    fn start_query<R>(
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
}

pub(super) fn try_mark_green<'tcx>(tcx: TyCtxt<'tcx>, dep_node: &dep_graph::DepNode) -> bool {
    tcx.dep_graph.try_mark_green(QueryCtxt::new(tcx), dep_node).is_some()
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
    if tcx.sess().opts.unstable_opts.incremental_verify_ich || cfg!(debug_assertions) {
        tcx.sess.time("query_key_hash_verify_all", || {
            for verify in super::QUERY_KEY_HASH_VERIFY.iter() {
                verify(tcx);
            }
        })
    }
}

macro_rules! cycle_error_handling {
    ([]) => {{
        rustc_query_system::query::CycleErrorHandling::Error
    }};
    ([(cycle_fatal) $($rest:tt)*]) => {{
        rustc_query_system::query::CycleErrorHandling::Fatal
    }};
    ([(cycle_stash) $($rest:tt)*]) => {{
        rustc_query_system::query::CycleErrorHandling::Stash
    }};
    ([(cycle_delay_bug) $($rest:tt)*]) => {{
        rustc_query_system::query::CycleErrorHandling::DelayBug
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
            let result = ::rustc_middle::query::erase::restore_val::<$V>(*result);
            ::rustc_query_system::dep_graph::hash_result(hcx, &result)
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

macro_rules! should_ever_cache_on_disk {
    ([]$yes:tt $no:tt) => {{
        $no
    }};
    ([(cache) $($rest:tt)*]$yes:tt $no:tt) => {{
        $yes
    }};
    ([$other:tt $($modifiers:tt)*]$yes:tt $no:tt) => {
        should_ever_cache_on_disk!([$($modifiers)*]$yes $no)
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
    Cache::Key: Key + DynSend + DynSync + for<'a> HashStable<StableHashingContext<'a>> + 'tcx,
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

pub(crate) fn encode_query_results<'a, 'tcx, Q>(
    query: Q::Dispatcher,
    qcx: QueryCtxt<'tcx>,
    encoder: &mut CacheEncoder<'a, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) where
    Q: QueryDispatcherUnerased<'tcx>,
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

fn all_inactive<'tcx, K>(state: &QueryState<'tcx, K>) -> bool {
    state.active.lock_shards().all(|shard| shard.is_empty())
}

pub(crate) fn query_key_hash_verify<'tcx>(
    query: impl QueryDispatcher<'tcx, Qcx = QueryCtxt<'tcx>>,
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

fn try_load_from_on_disk_cache<'tcx, Q>(query: Q, tcx: TyCtxt<'tcx>, dep_node: DepNode)
where
    Q: QueryDispatcher<'tcx, Qcx = QueryCtxt<'tcx>>,
{
    debug_assert!(tcx.dep_graph.is_green(&dep_node));

    let key = Q::Key::recover(tcx, &dep_node).unwrap_or_else(|| {
        panic!("Failed to recover key for {:?} with hash {}", dep_node, dep_node.hash)
    });
    if query.will_cache_on_disk_for_key(tcx, &key) {
        let _ = query.execute_query(tcx, key);
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

fn force_from_dep_node<'tcx, Q>(query: Q, tcx: TyCtxt<'tcx>, dep_node: DepNode) -> bool
where
    Q: QueryDispatcher<'tcx, Qcx = QueryCtxt<'tcx>>,
{
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

    if let Some(key) = Q::Key::recover(tcx, &dep_node) {
        force_query(query, QueryCtxt::new(tcx), key, dep_node);
        true
    } else {
        false
    }
}

pub(crate) fn make_dep_kind_vtable_for_query<'tcx, Q>(
    is_anon: bool,
    is_eval_always: bool,
) -> DepKindVTable<'tcx>
where
    Q: QueryDispatcherUnerased<'tcx>,
{
    let fingerprint_style = if is_anon {
        FingerprintStyle::Opaque
    } else {
        <Q::Dispatcher as QueryDispatcher>::Key::fingerprint_style()
    };

    if is_anon || !fingerprint_style.reconstructible() {
        return DepKindVTable {
            is_anon,
            is_eval_always,
            fingerprint_style,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
            name: Q::NAME,
        };
    }

    DepKindVTable {
        is_anon,
        is_eval_always,
        fingerprint_style,
        force_from_dep_node: Some(|tcx, dep_node, _| {
            force_from_dep_node(Q::query_dispatcher(tcx), tcx, dep_node)
        }),
        try_load_from_on_disk_cache: Some(|tcx, dep_node| {
            try_load_from_on_disk_cache(Q::query_dispatcher(tcx), tcx, dep_node)
        }),
        name: Q::NAME,
    }
}

macro_rules! item_if_cached {
    ([] $tokens:tt) => {};
    ([(cache) $($rest:tt)*] { $($tokens:tt)* }) => {
        $($tokens)*
    };
    ([$other:tt $($modifiers:tt)*] $tokens:tt) => {
        item_if_cached! { [$($modifiers)*] $tokens }
    };
}

macro_rules! expand_if_cached {
    ([], $tokens:expr) => {{
        None
    }};
    ([(cache) $($rest:tt)*], $tokens:expr) => {{
        Some($tokens)
    }};
    ([$other:tt $($modifiers:tt)*], $tokens:expr) => {
        expand_if_cached!([$($modifiers)*], $tokens)
    };
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

                // Adding `__rust_end_short_backtrace` marker to backtraces so that we emit the
                // frames when `RUST_BACKTRACE=1`, add a new mod with `$name` here is to allow
                // duplicate naming
                #[inline(never)]
                pub(crate) fn __rust_end_short_backtrace<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    span: Span,
                    key: queries::$name::Key<'tcx>,
                    mode: QueryMode,
                ) -> Option<Erased<queries::$name::Value<'tcx>>> {
                    #[cfg(debug_assertions)]
                    let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();
                    plumbing::get_query_incr(
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
                    Some(plumbing::get_query_non_incr(
                        QueryType::query_dispatcher(tcx),
                        QueryCtxt::new(tcx),
                        span,
                        key,
                    ))
                }
            }

            /// Defines a `compute` function for this query, to be used as a
            /// function pointer in the query's vtable.
            mod compute_fn {
                use super::*;
                use ::rustc_middle::queries::$name::{Key, Value, provided_to_erased};

                /// This function would be named `compute`, but we also want it
                /// to mark the boundaries of an omitted region in backtraces.
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
                    will_cache_on_disk_for_key_fn: should_ever_cache_on_disk!([$($modifiers)*] {
                        Some(queries::cached::$name)
                    } {
                        None
                    }),
                    execute_query: |tcx, key| erase::erase_val(tcx.$name(key)),
                    compute_fn: self::compute_fn::__rust_begin_short_backtrace,
                    try_load_from_disk_fn: should_ever_cache_on_disk!([$($modifiers)*] {
                        Some(|tcx, key, prev_index, index| {
                            // Check the `cache_on_disk_if` condition for this key.
                            if !queries::cached::$name(tcx, key) {
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
                    is_loadable_from_disk_fn: should_ever_cache_on_disk!([$($modifiers)*] {
                        Some(|tcx, key, index| -> bool {
                            ::rustc_middle::queries::cached::$name(tcx, key) &&
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

            impl<'tcx> QueryDispatcherUnerased<'tcx> for QueryType<'tcx> {
                type UnerasedValue = queries::$name::Value<'tcx>;
                type Dispatcher = SemiDynamicQueryDispatcher<
                    'tcx,
                    queries::$name::Storage<'tcx>,
                    FLAGS,
                >;

                const NAME: &'static &'static str = &stringify!($name);

                #[inline(always)]
                fn query_dispatcher(tcx: TyCtxt<'tcx>) -> Self::Dispatcher {
                    SemiDynamicQueryDispatcher {
                        vtable: &tcx.query_system.query_vtables.$name,
                    }
                }

                #[inline(always)]
                fn restore_val(value: <Self::Dispatcher as QueryDispatcher<'tcx>>::Value) -> Self::UnerasedValue {
                    erase::restore_val::<queries::$name::Value<'tcx>>(value)
                }
            }

            /// Internal per-query plumbing for collecting the set of active jobs for this query.
            ///
            /// Should only be called through `PER_QUERY_GATHER_ACTIVE_JOBS_FNS`.
            pub(crate) fn gather_active_jobs<'tcx>(
                tcx: TyCtxt<'tcx>,
                qmap: &mut QueryMap<'tcx>,
                require_complete: bool,
            ) -> Option<()> {
                let make_frame = |tcx: TyCtxt<'tcx>, key| {
                    let vtable = &tcx.query_system.query_vtables.$name;
                    $crate::plumbing::create_deferred_query_stack_frame(tcx, vtable, key)
                };

                // Call `gather_active_jobs_inner` to do the actual work.
                let res = plumbing::gather_active_jobs_inner(&tcx.query_system.states.$name,
                    tcx,
                    make_frame,
                    qmap,
                    require_complete,
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

            item_if_cached! { [$($modifiers)*] {
                pub(crate) fn encode_query_results<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    encoder: &mut CacheEncoder<'_, 'tcx>,
                    query_result_index: &mut EncodedDepNodeIndex
                ) {
                    $crate::plumbing::encode_query_results::<query_impl::$name::QueryType<'tcx>>(
                        query_impl::$name::QueryType::query_dispatcher(tcx),
                        QueryCtxt::new(tcx),
                        encoder,
                        query_result_index,
                    )
                }
            }}

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
            for<'tcx> fn(TyCtxt<'tcx>, &mut QueryMap<'tcx>, require_complete: bool) -> Option<()>
        ] = &[
            $(query_impl::$name::gather_active_jobs),*
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
        ] = &[$(expand_if_cached!([$($modifiers)*], query_impl::$name::encode_query_results)),*];

        const QUERY_KEY_HASH_VERIFY: &[
            for<'tcx> fn(TyCtxt<'tcx>)
        ] = &[$(query_impl::$name::query_key_hash_verify),*];

        /// Module containing a named function for each dep kind (including queries)
        /// that creates a `DepKindVTable`.
        ///
        /// Consumed via `make_dep_kind_array!` to create a list of vtables.
        #[expect(non_snake_case)]
        mod _dep_kind_vtable_ctors {
            use super::*;
            use rustc_middle::bug;
            use rustc_query_system::dep_graph::FingerprintStyle;

            // We use this for most things when incr. comp. is turned off.
            pub(crate) fn Null<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|_, dep_node, _| bug!("force_from_dep_node: encountered {:?}", dep_node)),
                    try_load_from_on_disk_cache: None,
                    name: &"Null",
                }
            }

            // We use this for the forever-red node.
            pub(crate) fn Red<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|_, dep_node, _| bug!("force_from_dep_node: encountered {:?}", dep_node)),
                    try_load_from_on_disk_cache: None,
                    name: &"Red",
                }
            }

            pub(crate) fn SideEffect<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|tcx, _, prev_index| {
                        tcx.dep_graph.force_diagnostic_node(QueryCtxt::new(tcx), prev_index);
                        true
                    }),
                    try_load_from_on_disk_cache: None,
                    name: &"SideEffect",
                }
            }

            pub(crate) fn AnonZeroDeps<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: true,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: Some(|_, _, _| bug!("cannot force an anon node")),
                    try_load_from_on_disk_cache: None,
                    name: &"AnonZeroDeps",
                }
            }

            pub(crate) fn TraitSelect<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: true,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"TraitSelect",
                }
            }

            pub(crate) fn CompileCodegenUnit<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"CompileCodegenUnit",
                }
            }

            pub(crate) fn CompileMonoItem<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"CompileMonoItem",
                }
            }

            pub(crate) fn Metadata<'tcx>() -> DepKindVTable<'tcx> {
                DepKindVTable {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"Metadata",
                }
            }

            $(pub(crate) fn $name<'tcx>() -> DepKindVTable<'tcx> {
                use $crate::query_impl::$name::QueryType;
                $crate::plumbing::make_dep_kind_vtable_for_query::<QueryType<'tcx>>(
                    is_anon!([$($modifiers)*]),
                    is_eval_always!([$($modifiers)*]),
                )
            })*
        }

        pub fn make_dep_kind_vtables<'tcx>(arena: &'tcx Arena<'tcx>) -> &'tcx [DepKindVTable<'tcx>] {
            arena.alloc_from_iter(rustc_middle::make_dep_kind_array!(_dep_kind_vtable_ctors))
        }
    }
}

#[inline(never)]
fn try_execute_query<'tcx, Q, const INCR: bool>(
    query: Q,
    qcx: Q::Qcx,
    span: Span,
    key: Q::Key,
    dep_node: Option<DepNode>,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryDispatcher<'tcx>,
{
    let state = query.query_state(qcx);
    let key_hash = sharded::make_hash(&key);
    let mut state_lock = state.active.lock_shard_by_hash(key_hash);

    // For the parallel compiler we need to check both the query cache and query state structures
    // while holding the state lock to ensure that 1) the query has not yet completed and 2) the
    // query is not still executing. Without checking the query cache here, we can end up
    // re-executing the query since `try_start` only checks that the query is not currently
    // executing, but another thread may have already completed the query and stores it result
    // in the query cache.
    if qcx.dep_context().sess().threads() > 1 {
        if let Some((value, index)) = query.query_cache(qcx).lookup(&key) {
            qcx.dep_context().profiler().query_cache_hit(index.into());
            return (value, Some(index));
        }
    }

    let current_job_id = qcx.current_query_job();

    match state_lock.entry(key_hash, equivalent_key(&key), |(k, _)| sharded::make_hash(k)) {
        Entry::Vacant(entry) => {
            // Nothing has computed or is computing the query, so we start a new job and insert it
            // in the state map.
            let id = qcx.next_job_id();
            let job = QueryJob::new(id, span, current_job_id);
            entry.insert((key, ActiveKeyStatus::Started(job)));

            // Drop the lock before we start executing the query
            drop(state_lock);

            execute_job::<Q, INCR>(query, qcx, state, key, key_hash, id, dep_node)
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
                        return wait_for_query(query, qcx, span, key, latch, current_job_id);
                    }

                    let id = job.id;
                    drop(state_lock);

                    // If we are single-threaded we know that we have cycle error,
                    // so we just return the error.
                    cycle_error(query, qcx, id, span)
                }
                ActiveKeyStatus::Poisoned => FatalError.raise(),
            }
        }
    }
}

#[inline(always)]
pub(super) fn get_query_non_incr<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    span: Span,
    key: Q::Key,
) -> Q::Value
where
    Q: QueryDispatcher<'tcx>,
{
    debug_assert!(!qcx.dep_context().dep_graph().is_fully_enabled());

    ensure_sufficient_stack(|| try_execute_query::<Q, false>(query, qcx, span, key, None).0)
}

#[inline(always)]
pub(super) fn get_query_incr<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    span: Span,
    key: Q::Key,
    mode: QueryMode,
) -> Option<Q::Value>
where
    Q: QueryDispatcher<'tcx>,
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

    let (result, dep_node_index) =
        ensure_sufficient_stack(|| try_execute_query::<Q, true>(query, qcx, span, key, dep_node));
    if let Some(dep_node_index) = dep_node_index {
        qcx.dep_context().dep_graph().read_index(dep_node_index)
    }
    Some(result)
}

fn force_query<'tcx, Q>(query: Q, qcx: Q::Qcx, key: Q::Key, dep_node: DepNode)
where
    Q: QueryDispatcher<'tcx>,
{
    // We may be concurrently trying both execute and force a query.
    // Ensure that only one of them runs the query.
    if let Some((_, index)) = query.query_cache(qcx).lookup(&key) {
        qcx.dep_context().profiler().query_cache_hit(index.into());
        return;
    }

    debug_assert!(!query.anon());

    ensure_sufficient_stack(|| {
        try_execute_query::<Q, true>(query, qcx, DUMMY_SP, key, Some(dep_node))
    });
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
fn ensure_must_run<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    key: &Q::Key,
    check_cache: bool,
) -> (bool, Option<DepNode>)
where
    Q: QueryDispatcher<'tcx>,
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

    let loadable = query.is_loadable_from_disk(qcx, key, serialized_dep_node_index);
    (!loadable, Some(dep_node))
}

#[cold]
#[inline(never)]
fn cycle_error<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    try_execute: QueryJobId,
    span: Span,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryDispatcher<'tcx>,
{
    // Ensure there was no errors collecting all active jobs.
    // We need the complete map to ensure we find a cycle to break.
    let query_map = qcx
        .collect_active_jobs_from_all_queries(false)
        .ok()
        .expect("failed to collect active queries");

    let error = try_execute.find_cycle_in_stack(query_map, &qcx.current_query_job(), span);
    (mk_cycle(query, qcx, error.lift()), None)
}

#[inline(always)]
fn wait_for_query<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    span: Span,
    key: Q::Key,
    latch: QueryLatch<'tcx>,
    current: Option<QueryJobId>,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryDispatcher<'tcx>,
{
    // For parallel queries, we'll block and wait until the query running
    // in another thread has completed. Record how long we wait in the
    // self-profiler.
    let query_blocked_prof_timer = qcx.dep_context().profiler().query_blocked();

    // With parallel queries we might just have to wait on some other
    // thread.
    let result = latch.wait_on(qcx, current, span);

    match result {
        Ok(()) => {
            let Some((v, index)) = query.query_cache(qcx).lookup(&key) else {
                outline(|| {
                    // We didn't find the query result in the query cache. Check if it was
                    // poisoned due to a panic instead.
                    let key_hash = sharded::make_hash(&key);
                    let shard = query.query_state(qcx).active.lock_shard_by_hash(key_hash);
                    match shard.find(key_hash, equivalent_key(&key)) {
                        // The query we waited on panicked. Continue unwinding here.
                        Some((_, ActiveKeyStatus::Poisoned)) => FatalError.raise(),
                        _ => panic!(
                            "query '{}' result must be in the cache or the query must be poisoned after a wait",
                            query.name()
                        ),
                    }
                })
            };

            qcx.dep_context().profiler().query_cache_hit(index.into());
            query_blocked_prof_timer.finish_with_query_invocation_id(index.into());

            (v, Some(index))
        }
        Err(cycle) => (mk_cycle(query, qcx, cycle.lift()), None),
    }
}

#[cold]
#[inline(never)]
fn mk_cycle<'tcx, Q>(query: Q, qcx: Q::Qcx, cycle_error: CycleError) -> Q::Value
where
    Q: QueryDispatcher<'tcx>,
{
    let error = report_cycle(qcx.dep_context().sess(), &cycle_error);
    handle_cycle_error(query, qcx, &cycle_error, error)
}

fn handle_cycle_error<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    cycle_error: &CycleError,
    error: Diag<'_>,
) -> Q::Value
where
    Q: QueryDispatcher<'tcx>,
{
    match query.cycle_error_handling() {
        CycleErrorHandling::Error => {
            let guar = error.emit();
            query.value_from_cycle_error(*qcx.dep_context(), cycle_error, guar)
        }
        CycleErrorHandling::Fatal => {
            error.emit();
            qcx.dep_context().sess().dcx().abort_if_errors();
            unreachable!()
        }
        CycleErrorHandling::DelayBug => {
            let guar = error.delay_as_bug();
            query.value_from_cycle_error(*qcx.dep_context(), cycle_error, guar)
        }
        CycleErrorHandling::Stash => {
            let guar = if let Some(root) = cycle_error.cycle.first()
                && let Some(span) = root.frame.info.span
            {
                error.stash(span, StashKey::Cycle).unwrap()
            } else {
                error.emit()
            };
            query.value_from_cycle_error(*qcx.dep_context(), cycle_error, guar)
        }
    }
}

#[inline(always)]
fn execute_job<'tcx, Q, const INCR: bool>(
    query: Q,
    qcx: Q::Qcx,
    state: &'tcx QueryState<'tcx, Q::Key>,
    key: Q::Key,
    key_hash: u64,
    id: QueryJobId,
    dep_node: Option<DepNode>,
) -> (Q::Value, Option<DepNodeIndex>)
where
    Q: QueryDispatcher<'tcx>,
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
                // results is tainted by errors.
                assert!(
                    qcx.dep_context().sess().dcx().has_errors().is_some(),
                    "Computed query value for {:?}({:?}) is inconsistent with fed value,\n\
                        computed={:#?}\nfed={:#?}",
                    query.dep_kind(),
                    key,
                    formatter(&result),
                    formatter(&cached_result),
                );
            }
        }
    }
    job_owner.complete(cache, key_hash, result, dep_node_index);

    (result, Some(dep_node_index))
}

// Fast path for when incr. comp. is off.
#[inline(always)]
fn execute_job_non_incr<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    key: Q::Key,
    job_id: QueryJobId,
) -> (Q::Value, DepNodeIndex)
where
    Q: QueryDispatcher<'tcx>,
{
    debug_assert!(!qcx.dep_context().dep_graph().is_fully_enabled());

    // Fingerprint the key, just to assert that it doesn't
    // have anything we don't consider hashable
    if cfg!(debug_assertions) {
        let _ = key.to_fingerprint(*qcx.dep_context());
    }

    let prof_timer = qcx.dep_context().profiler().query_provider();
    let result = qcx.start_query(job_id, query.depth_limit(), || query.compute(qcx, key));
    let dep_node_index = qcx.dep_context().dep_graph().next_virtual_depnode_index();
    prof_timer.finish_with_query_invocation_id(dep_node_index.into());

    // Similarly, fingerprint the result to assert that
    // it doesn't have anything not considered hashable.
    if cfg!(debug_assertions)
        && let Some(hash_result) = query.hash_result()
    {
        qcx.dep_context().with_stable_hashing_context(|mut hcx| {
            hash_result(&mut hcx, &result);
        });
    }

    (result, dep_node_index)
}

#[inline(always)]
fn execute_job_incr<'tcx, Q>(
    query: Q,
    qcx: Q::Qcx,
    dep_graph_data: &DepGraphData<<Q::Qcx as HasDepContext>::Deps>,
    key: Q::Key,
    mut dep_node_opt: Option<DepNode>,
    job_id: QueryJobId,
) -> (Q::Value, DepNodeIndex)
where
    Q: QueryDispatcher<'tcx>,
{
    if !query.anon() && !query.eval_always() {
        // `to_dep_node` is expensive for some `DepKind`s.
        let dep_node =
            dep_node_opt.get_or_insert_with(|| query.construct_dep_node(*qcx.dep_context(), &key));

        // The diagnostics for this query will be promoted to the current session during
        // `try_mark_green()`, so we can ignore them here.
        if let Some(ret) = qcx.start_query(job_id, false, || {
            try_load_from_disk_and_cache_in_memory(query, dep_graph_data, qcx, &key, dep_node)
        }) {
            return ret;
        }
    }

    let prof_timer = qcx.dep_context().profiler().query_provider();

    let (result, dep_node_index) = qcx.start_query(job_id, query.depth_limit(), || {
        if query.anon() {
            return dep_graph_data.with_anon_task_inner(
                *qcx.dep_context(),
                query.dep_kind(),
                || query.compute(qcx, key),
            );
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

    (result, dep_node_index)
}

#[inline(always)]
fn try_load_from_disk_and_cache_in_memory<'tcx, Q>(
    query: Q,
    dep_graph_data: &DepGraphData<<Q::Qcx as HasDepContext>::Deps>,
    qcx: Q::Qcx,
    key: &Q::Key,
    dep_node: &DepNode,
) -> Option<(Q::Value, DepNodeIndex)>
where
    Q: QueryDispatcher<'tcx>,
{
    // Note this function can be called concurrently from the same query
    // We must ensure that this is handled correctly.

    let (prev_dep_node_index, dep_node_index) = dep_graph_data.try_mark_green(qcx, dep_node)?;

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
        let try_verify = prev_fingerprint.split().1.as_u64().is_multiple_of(32);
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
        !query.will_cache_on_disk_for_key(*qcx.dep_context(), key)
            || !qcx.dep_context().fingerprint_style(dep_node.kind).reconstructible(),
        "missing on-disk cache entry for {dep_node:?}"
    );

    // Sanity check for the logic in `ensure`: if the node is green and the result loadable,
    // we should actually be able to load it.
    debug_assert!(
        !query.is_loadable_from_disk(qcx, key, prev_dep_node_index),
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

/// A type representing the responsibility to execute the job in the `job` field.
/// This will poison the relevant query if dropped.
struct JobOwner<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    state: &'tcx QueryState<'tcx, K>,
    key: K,
}

impl<'tcx, K> JobOwner<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    /// Completes the query by updating the query cache with the `result`,
    /// signals the waiter and forgets the JobOwner, so it won't poison the query
    fn complete<C>(self, cache: &C, key_hash: u64, result: C::Value, dep_node_index: DepNodeIndex)
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
            // don't keep the lock during the `unwrap()` of the retrieved value, or we taint the
            // underlying shard.
            // since unwinding also wants to look at this map, this can also prevent a double
            // panic.
            let mut shard = state.active.lock_shard_by_hash(key_hash);
            match shard.find_entry(key_hash, equivalent_key(&key)) {
                Err(_) => None,
                Ok(occupied) => Some(occupied.remove().0.1),
            }
        };
        let job = expect_job(job.expect("active query job entry"));

        job.signal_complete();
    }
}

impl<'tcx, K> Drop for JobOwner<'tcx, K>
where
    K: Eq + Hash + Copy,
{
    #[inline(never)]
    #[cold]
    fn drop(&mut self) {
        // Poison the query so jobs waiting on it panic.
        let state = self.state;
        let job = {
            let key_hash = sharded::make_hash(&self.key);
            let mut shard = state.active.lock_shard_by_hash(key_hash);
            match shard.find_entry(key_hash, equivalent_key(&self.key)) {
                Err(_) => panic!(),
                Ok(occupied) => {
                    let ((key, value), vacant) = occupied.remove();
                    vacant.insert((key, ActiveKeyStatus::Poisoned));
                    expect_job(value)
                }
            }
        };
        // Also signal the completion of the job, so waiters
        // will continue execution.
        job.signal_complete();
    }
}

#[inline]
fn equivalent_key<K: Eq, V>(k: &K) -> impl Fn(&(K, V)) -> bool + '_ {
    move |x| x.0 == *k
}

/// Obtains the enclosed [`QueryJob`], or panics if this query evaluation
/// was poisoned by a panic.
fn expect_job<'tcx>(status: ActiveKeyStatus<'tcx>) -> QueryJob<'tcx> {
    match status {
        ActiveKeyStatus::Started(job) => job,
        ActiveKeyStatus::Poisoned => {
            panic!("job for query failed to start and was poisoned")
        }
    }
}

/// Internal plumbing for collecting the set of active jobs for this query.
///
/// Should only be called from `gather_active_jobs`.
pub(crate) fn gather_active_jobs_inner<'tcx, Qcx: Copy, K: Copy>(
    state: &QueryState<'tcx, K>,
    qcx: Qcx,
    make_frame: fn(Qcx, K) -> QueryStackFrame<QueryStackDeferred<'tcx>>,
    jobs: &mut QueryMap<'tcx>,
    require_complete: bool,
) -> Option<()> {
    let mut active = Vec::new();

    // Helper to gather active jobs from a single shard.
    let mut gather_shard_jobs = |shard: &HashTable<(K, ActiveKeyStatus<'tcx>)>| {
        for (k, v) in shard.iter() {
            if let ActiveKeyStatus::Started(ref job) = *v {
                active.push((*k, job.clone()));
            }
        }
    };

    // Lock shards and gather jobs from each shard.
    if require_complete {
        for shard in state.active.lock_shards() {
            gather_shard_jobs(&shard);
        }
    } else {
        // We use try_lock_shards here since we are called from the
        // deadlock handler, and this shouldn't be locked.
        for shard in state.active.try_lock_shards() {
            let shard = shard?;
            gather_shard_jobs(&shard);
        }
    }

    // Call `make_frame` while we're not holding a `state.active` lock as `make_frame` may call
    // queries leading to a deadlock.
    for (key, job) in active {
        let frame = make_frame(qcx, key);
        jobs.insert(job.id, QueryJobInfo { frame, job });
    }

    Some(())
}
