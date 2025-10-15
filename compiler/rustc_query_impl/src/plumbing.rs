//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use std::num::NonZero;

use rustc_data_structures::jobserver::Proxy;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_data_structures::unord::UnordMap;
use rustc_hashes::Hash64;
use rustc_hir::limit::Limit;
use rustc_index::Idx;
use rustc_middle::bug;
use rustc_middle::dep_graph::{
    self, DepContext, DepKind, DepKindStruct, DepNode, DepNodeIndex, SerializedDepNodeIndex,
    dep_kinds,
};
use rustc_middle::query::Key;
use rustc_middle::query::on_disk_cache::{
    AbsoluteBytePos, CacheDecoder, CacheEncoder, EncodedDepNodeIndex,
};
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::print::with_reduced_queries;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::ty::{self, TyCtxt};
use rustc_query_system::dep_graph::{DepNodeParams, HasDepContext};
use rustc_query_system::ich::StableHashingContext;
use rustc_query_system::query::{
    QueryCache, QueryConfig, QueryContext, QueryJobId, QueryMap, QuerySideEffect,
    QueryStackDeferred, QueryStackFrame, QueryStackFrameExtra, force_query,
};
use rustc_query_system::{QueryOverflow, QueryOverflowNote};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::def_id::LOCAL_CRATE;

use crate::QueryConfigRestored;

#[derive(Copy, Clone)]
pub struct QueryCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

impl<'tcx> QueryCtxt<'tcx> {
    #[inline]
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        QueryCtxt { tcx }
    }
}

impl<'tcx> std::ops::Deref for QueryCtxt<'tcx> {
    type Target = TyCtxt<'tcx>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.tcx
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

impl<'tcx> QueryContext for QueryCtxt<'tcx> {
    type QueryInfo = QueryStackDeferred<'tcx>;

    #[inline]
    fn jobserver_proxy(&self) -> &Proxy {
        &*self.jobserver_proxy
    }

    #[inline]
    fn next_job_id(self) -> QueryJobId {
        QueryJobId(
            NonZero::new(self.query_system.jobs.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                .unwrap(),
        )
    }

    #[inline]
    fn current_query_job(self) -> Option<QueryJobId> {
        tls::with_related_context(self.tcx, |icx| icx.query)
    }

    /// Returns a query map representing active query jobs.
    /// It returns an incomplete map as an error if it fails
    /// to take locks.
    fn collect_active_jobs(
        self,
    ) -> Result<QueryMap<QueryStackDeferred<'tcx>>, QueryMap<QueryStackDeferred<'tcx>>> {
        let mut jobs = QueryMap::default();
        let mut complete = true;

        for collect in super::TRY_COLLECT_ACTIVE_JOBS.iter() {
            if collect(self.tcx, &mut jobs).is_none() {
                complete = false;
            }
        }

        if complete { Ok(jobs) } else { Err(jobs) }
    }

    fn lift_query_info(
        self,
        info: &QueryStackDeferred<'tcx>,
    ) -> rustc_query_system::query::QueryStackFrameExtra {
        info.extract()
    }

    // Interactions with on_disk_cache
    fn load_side_effect(
        self,
        prev_dep_node_index: SerializedDepNodeIndex,
    ) -> Option<QuerySideEffect> {
        self.query_system
            .on_disk_cache
            .as_ref()
            .and_then(|c| c.load_side_effect(self.tcx, prev_dep_node_index))
    }

    #[inline(never)]
    #[cold]
    fn store_side_effect(self, dep_node_index: DepNodeIndex, side_effect: QuerySideEffect) {
        if let Some(c) = self.query_system.on_disk_cache.as_ref() {
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
            if depth_limit && !self.recursion_limit().value_within_limit(current_icx.query_depth) {
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

    fn depth_limit_error(self, job: QueryJobId) {
        // FIXME: `collect_active_jobs` expects no locks to be held, which doesn't hold for this call.
        let query_map = match self.collect_active_jobs() {
            Ok(query_map) => query_map,
            Err(query_map) => query_map,
        };
        let (info, depth) = job.find_dep_kind_root(query_map);

        let suggested_limit = match self.recursion_limit() {
            Limit(0) => Limit(2),
            limit => limit * 2,
        };

        self.sess.dcx().emit_fatal(QueryOverflow {
            span: info.job.span,
            note: QueryOverflowNote {
                desc: self.lift_query_info(&info.query.info).description,
                depth,
            },
            suggested_limit,
            crate_name: self.crate_name(LOCAL_CRATE),
        });
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

macro_rules! handle_cycle_error {
    ([]) => {{
        rustc_query_system::HandleCycleError::Error
    }};
    ([(fatal_cycle) $($rest:tt)*]) => {{
        rustc_query_system::HandleCycleError::Fatal
    }};
    ([(cycle_stash) $($rest:tt)*]) => {{
        rustc_query_system::HandleCycleError::Stash
    }};
    ([(cycle_delay_bug) $($rest:tt)*]) => {{
        rustc_query_system::HandleCycleError::DelayBug
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        handle_cycle_error!([$($modifiers)*])
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
        Some(|hcx, result| dep_graph::hash_result(hcx, &restore::<$V>(*result)))
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

fn create_query_frame_extra<'tcx, K: Key + Copy + 'tcx>(
    (tcx, key, kind, name, do_describe): (
        TyCtxt<'tcx>,
        K,
        DepKind,
        &'static str,
        fn(TyCtxt<'tcx>, K) -> String,
    ),
) -> QueryStackFrameExtra {
    let def_id = key.key_as_def_id();

    // If reduced queries are requested, we may be printing a query stack due
    // to a panic. Avoid using `default_span` and `def_kind` in that case.
    let reduce_queries = with_reduced_queries();

    // Avoid calling queries while formatting the description
    let description = ty::print::with_no_queries!(do_describe(tcx, key));
    let description = if tcx.sess.verbose_internals() {
        format!("{description} [{name:?}]")
    } else {
        description
    };
    let span = if kind == dep_graph::dep_kinds::def_span || reduce_queries {
        // The `def_span` query is used to calculate `default_span`,
        // so exit to avoid infinite recursion.
        None
    } else {
        Some(key.default_span(tcx))
    };

    let def_kind = if kind == dep_graph::dep_kinds::def_kind || reduce_queries {
        // Try to avoid infinite recursion.
        None
    } else {
        def_id.and_then(|def_id| def_id.as_local()).map(|def_id| tcx.def_kind(def_id))
    };
    QueryStackFrameExtra::new(description, span, def_kind)
}

pub(crate) fn create_query_frame<
    'tcx,
    K: Copy + DynSend + DynSync + Key + for<'a> HashStable<StableHashingContext<'a>> + 'tcx,
>(
    tcx: TyCtxt<'tcx>,
    do_describe: fn(TyCtxt<'tcx>, K) -> String,
    key: K,
    kind: DepKind,
    name: &'static str,
) -> QueryStackFrame<QueryStackDeferred<'tcx>> {
    let def_id = key.key_as_def_id();

    let hash = || {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            kind.as_usize().hash_stable(&mut hcx, &mut hasher);
            key.hash_stable(&mut hcx, &mut hasher);
            hasher.finish::<Hash64>()
        })
    };
    let def_id_for_ty_in_cycle = key.def_id_for_ty_in_cycle();

    let info =
        QueryStackDeferred::new((tcx, key, kind, name, do_describe), create_query_frame_extra);

    QueryStackFrame::new(info, kind, hash, def_id, def_id_for_ty_in_cycle)
}

pub(crate) fn encode_query_results<'a, 'tcx, Q>(
    query: Q::Config,
    qcx: QueryCtxt<'tcx>,
    encoder: &mut CacheEncoder<'a, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) where
    Q: super::QueryConfigRestored<'tcx>,
    Q::RestoredValue: Encodable<CacheEncoder<'a, 'tcx>>,
{
    let _timer = qcx.profiler().generic_activity_with_arg("encode_query_results_for", query.name());

    assert!(query.query_state(qcx).all_inactive());
    let cache = query.query_cache(qcx);
    cache.iter(&mut |key, value, dep_node| {
        if query.cache_on_disk(qcx.tcx, key) {
            let dep_node = SerializedDepNodeIndex::new(dep_node.index());

            // Record position of the cache entry.
            query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.position())));

            // Encode the type check tables with the `SerializedDepNodeIndex`
            // as tag.
            encoder.encode_tagged(dep_node, &Q::restore(*value));
        }
    });
}

pub(crate) fn query_key_hash_verify<'tcx>(
    query: impl QueryConfig<QueryCtxt<'tcx>>,
    qcx: QueryCtxt<'tcx>,
) {
    let _timer =
        qcx.profiler().generic_activity_with_arg("query_key_hash_verify_for", query.name());

    let mut map = UnordMap::default();

    let cache = query.query_cache(qcx);
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
    Q: QueryConfig<QueryCtxt<'tcx>>,
{
    debug_assert!(tcx.dep_graph.is_green(&dep_node));

    let key = Q::Key::recover(tcx, &dep_node).unwrap_or_else(|| {
        panic!("Failed to recover key for {:?} with hash {}", dep_node, dep_node.hash)
    });
    if query.cache_on_disk(tcx, &key) {
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
    Q: QueryConfig<QueryCtxt<'tcx>>,
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

pub(crate) fn query_callback<'tcx, Q>(is_anon: bool, is_eval_always: bool) -> DepKindStruct<'tcx>
where
    Q: QueryConfigRestored<'tcx>,
{
    let fingerprint_style = <Q::Config as QueryConfig<QueryCtxt<'tcx>>>::Key::fingerprint_style();

    if is_anon || !fingerprint_style.reconstructible() {
        return DepKindStruct {
            is_anon,
            is_eval_always,
            fingerprint_style,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
            name: Q::NAME,
        };
    }

    DepKindStruct {
        is_anon,
        is_eval_always,
        fingerprint_style,
        force_from_dep_node: Some(|tcx, dep_node, _| {
            force_from_dep_node(Q::config(tcx), tcx, dep_node)
        }),
        try_load_from_on_disk_cache: Some(|tcx, dep_node| {
            try_load_from_on_disk_cache(Q::config(tcx), tcx, dep_node)
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

/// Don't show the backtrace for query system by default
/// use `RUST_BACKTRACE=full` to show all the backtraces
#[inline(never)]
pub(crate) fn __rust_begin_short_backtrace<F, T>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let result = f();
    std::hint::black_box(());
    result
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
                ) -> Option<Erase<queries::$name::Value<'tcx>>> {
                    #[cfg(debug_assertions)]
                    let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();
                    get_query_incr(
                        QueryType::config(tcx),
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
                ) -> Option<Erase<queries::$name::Value<'tcx>>> {
                    Some(get_query_non_incr(
                        QueryType::config(tcx),
                        QueryCtxt::new(tcx),
                        span,
                        key,
                    ))
                }
            }

            pub(crate) fn dynamic_query<'tcx>()
                -> DynamicQuery<'tcx, queries::$name::Storage<'tcx>>
            {
                DynamicQuery {
                    name: stringify!($name),
                    eval_always: is_eval_always!([$($modifiers)*]),
                    dep_kind: dep_graph::dep_kinds::$name,
                    handle_cycle_error: handle_cycle_error!([$($modifiers)*]),
                    query_state: std::mem::offset_of!(QueryStates<'tcx>, $name),
                    query_cache: std::mem::offset_of!(QueryCaches<'tcx>, $name),
                    cache_on_disk: |tcx, key| ::rustc_middle::query::cached::$name(tcx, key),
                    execute_query: |tcx, key| erase(tcx.$name(key)),
                    compute: |tcx, key| {
                        #[cfg(debug_assertions)]
                        let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();
                        __rust_begin_short_backtrace(||
                            queries::$name::provided_to_erased(
                                tcx,
                                {
                                    let ret = call_provider!([$($modifiers)*][tcx, $name, key]);
                                    rustc_middle::ty::print::with_reduced_queries!({
                                        tracing::trace!(?ret);
                                    });
                                    ret
                                }
                            )
                        )
                    },
                    can_load_from_disk: should_ever_cache_on_disk!([$($modifiers)*] true false),
                    try_load_from_disk: should_ever_cache_on_disk!([$($modifiers)*] {
                        |tcx, key, prev_index, index| {
                            if ::rustc_middle::query::cached::$name(tcx, key) {
                                let value = $crate::plumbing::try_load_from_disk::<
                                    queries::$name::ProvidedValue<'tcx>
                                >(
                                    tcx,
                                    prev_index,
                                    index,
                                );
                                value.map(|value| queries::$name::provided_to_erased(tcx, value))
                            } else {
                                None
                            }
                        }
                    } {
                        |_tcx, _key, _prev_index, _index| None
                    }),
                    value_from_cycle_error: |tcx, cycle, guar| {
                        let result: queries::$name::Value<'tcx> = Value::from_cycle_error(tcx, cycle, guar);
                        erase(result)
                    },
                    loadable_from_disk: |_tcx, _key, _index| {
                        should_ever_cache_on_disk!([$($modifiers)*] {
                            ::rustc_middle::query::cached::$name(_tcx, _key) &&
                                $crate::plumbing::loadable_from_disk(_tcx, _index)
                        } {
                            false
                        })
                    },
                    hash_result: hash_result!([$($modifiers)*][queries::$name::Value<'tcx>]),
                    format_value: |value| format!("{:?}", restore::<queries::$name::Value<'tcx>>(*value)),
                }
            }

            #[derive(Copy, Clone, Default)]
            pub(crate) struct QueryType<'tcx> {
                data: PhantomData<&'tcx ()>
            }

            impl<'tcx> QueryConfigRestored<'tcx> for QueryType<'tcx> {
                type RestoredValue = queries::$name::Value<'tcx>;
                type Config = DynamicConfig<
                    'tcx,
                    queries::$name::Storage<'tcx>,
                    { is_anon!([$($modifiers)*]) },
                    { depth_limit!([$($modifiers)*]) },
                    { feedable!([$($modifiers)*]) },
                >;

                const NAME: &'static &'static str = &stringify!($name);

                #[inline(always)]
                fn config(tcx: TyCtxt<'tcx>) -> Self::Config {
                    DynamicConfig {
                        dynamic: &tcx.query_system.dynamic_queries.$name,
                    }
                }

                #[inline(always)]
                fn restore(value: <Self::Config as QueryConfig<QueryCtxt<'tcx>>>::Value) -> Self::RestoredValue {
                    restore::<queries::$name::Value<'tcx>>(value)
                }
            }

            pub(crate) fn try_collect_active_jobs<'tcx>(
                tcx: TyCtxt<'tcx>,
                qmap: &mut QueryMap<QueryStackDeferred<'tcx>>,
            ) -> Option<()> {
                let make_query = |tcx, key| {
                    let kind = rustc_middle::dep_graph::dep_kinds::$name;
                    let name = stringify!($name);
                    $crate::plumbing::create_query_frame(tcx, rustc_middle::query::descs::$name, key, kind, name)
                };
                let res = tcx.query_system.states.$name.try_collect_active_jobs(
                    tcx,
                    make_query,
                    qmap,
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
                        query_impl::$name::QueryType::config(tcx),
                        QueryCtxt::new(tcx),
                        encoder,
                        query_result_index,
                    )
                }
            }}

            pub(crate) fn query_key_hash_verify<'tcx>(tcx: TyCtxt<'tcx>) {
                $crate::plumbing::query_key_hash_verify(
                    query_impl::$name::QueryType::config(tcx),
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

        pub fn dynamic_queries<'tcx>() -> DynamicQueries<'tcx> {
            DynamicQueries {
                $(
                    $name: query_impl::$name::dynamic_query(),
                )*
            }
        }

        // These arrays are used for iteration and can't be indexed by `DepKind`.

        const TRY_COLLECT_ACTIVE_JOBS: &[
            for<'tcx> fn(TyCtxt<'tcx>, &mut QueryMap<QueryStackDeferred<'tcx>>) -> Option<()>
        ] =
            &[$(query_impl::$name::try_collect_active_jobs),*];

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

        #[allow(nonstandard_style)]
        mod query_callbacks {
            use super::*;
            use rustc_middle::bug;
            use rustc_query_system::dep_graph::FingerprintStyle;

            // We use this for most things when incr. comp. is turned off.
            pub(crate) fn Null<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|_, dep_node, _| bug!("force_from_dep_node: encountered {:?}", dep_node)),
                    try_load_from_on_disk_cache: None,
                    name: &"Null",
                }
            }

            // We use this for the forever-red node.
            pub(crate) fn Red<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|_, dep_node, _| bug!("force_from_dep_node: encountered {:?}", dep_node)),
                    try_load_from_on_disk_cache: None,
                    name: &"Red",
                }
            }

            pub(crate) fn SideEffect<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
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

            pub(crate) fn AnonZeroDeps<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: true,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: Some(|_, _, _| bug!("cannot force an anon node")),
                    try_load_from_on_disk_cache: None,
                    name: &"AnonZeroDeps",
                }
            }

            pub(crate) fn TraitSelect<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: true,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"TraitSelect",
                }
            }

            pub(crate) fn CompileCodegenUnit<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"CompileCodegenUnit",
                }
            }

            pub(crate) fn CompileMonoItem<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"CompileMonoItem",
                }
            }

            pub(crate) fn Metadata<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                    name: &"Metadata",
                }
            }

            $(pub(crate) fn $name<'tcx>()-> DepKindStruct<'tcx> {
                $crate::plumbing::query_callback::<query_impl::$name::QueryType<'tcx>>(
                    is_anon!([$($modifiers)*]),
                    is_eval_always!([$($modifiers)*]),
                )
            })*
        }

        pub fn query_callbacks<'tcx>(arena: &'tcx Arena<'tcx>) -> &'tcx [DepKindStruct<'tcx>] {
            arena.alloc_from_iter(rustc_middle::make_dep_kind_array!(query_callbacks))
        }

        pub fn dep_kind_names() -> Vec<&'static str> {
            rustc_middle::make_dep_kind_name_array!(query_callbacks)
        }
    }
}
