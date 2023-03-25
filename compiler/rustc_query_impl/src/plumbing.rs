//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::rustc_middle::dep_graph::DepContext;
use crate::rustc_middle::ty::TyEncoder;
use rustc_data_structures::stable_hasher::{Hash64, HashStable, StableHasher};
use rustc_data_structures::sync::Lock;
use rustc_errors::Diagnostic;
use rustc_index::Idx;
use rustc_middle::dep_graph::{
    self, DepKind, DepKindStruct, DepNode, DepNodeIndex, SerializedDepNodeIndex,
};
use rustc_middle::query::on_disk_cache::AbsoluteBytePos;
use rustc_middle::query::on_disk_cache::{CacheDecoder, CacheEncoder, EncodedDepNodeIndex};
use rustc_middle::query::Key;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::ty::{self, TyCtxt};
use rustc_query_system::dep_graph::{DepNodeParams, HasDepContext};
use rustc_query_system::ich::StableHashingContext;
use rustc_query_system::query::{
    force_query, QueryCache, QueryConfig, QueryContext, QueryJobId, QueryMap, QuerySideEffects,
    QueryStackFrame,
};
use rustc_query_system::{LayoutOfDepth, QueryOverflow};
use rustc_serialize::Decodable;
use rustc_serialize::Encodable;
use rustc_session::Limit;
use rustc_span::def_id::LOCAL_CRATE;
use std::num::NonZeroU64;
use thin_vec::ThinVec;

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
    type DepKind = rustc_middle::dep_graph::DepKind;
    type DepContext = TyCtxt<'tcx>;

    #[inline]
    fn dep_context(&self) -> &Self::DepContext {
        &self.tcx
    }
}

impl QueryContext for QueryCtxt<'_> {
    #[inline]
    fn next_job_id(self) -> QueryJobId {
        QueryJobId(
            NonZeroU64::new(
                self.query_system.jobs.fetch_add(1, rustc_data_structures::sync::Ordering::Relaxed),
            )
            .unwrap(),
        )
    }

    #[inline]
    fn current_query_job(self) -> Option<QueryJobId> {
        tls::with_related_context(self.tcx, |icx| icx.query)
    }

    fn try_collect_active_jobs(self) -> Option<QueryMap<DepKind>> {
        let mut jobs = QueryMap::default();

        for query in &self.query_system.fns.query_structs {
            (query.try_collect_active_jobs)(self.tcx, &mut jobs);
        }

        Some(jobs)
    }

    // Interactions with on_disk_cache
    fn load_side_effects(self, prev_dep_node_index: SerializedDepNodeIndex) -> QuerySideEffects {
        self.query_system
            .on_disk_cache
            .as_ref()
            .map(|c| c.load_side_effects(self.tcx, prev_dep_node_index))
            .unwrap_or_default()
    }

    #[inline(never)]
    #[cold]
    fn store_side_effects(self, dep_node_index: DepNodeIndex, side_effects: QuerySideEffects) {
        if let Some(c) = self.query_system.on_disk_cache.as_ref() {
            c.store_side_effects(dep_node_index, side_effects)
        }
    }

    #[inline(never)]
    #[cold]
    fn store_side_effects_for_anon_node(
        self,
        dep_node_index: DepNodeIndex,
        side_effects: QuerySideEffects,
    ) {
        if let Some(c) = self.query_system.on_disk_cache.as_ref() {
            c.store_side_effects_for_anon_node(dep_node_index, side_effects)
        }
    }

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    #[inline(always)]
    fn start_query<R>(
        self,
        token: QueryJobId,
        depth_limit: bool,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
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
                diagnostics,
                query_depth: current_icx.query_depth + depth_limit as usize,
                task_deps: current_icx.task_deps,
            };

            // Use the `ImplicitCtxt` while we execute the query.
            tls::enter_context(&new_icx, compute)
        })
    }

    fn depth_limit_error(self, job: QueryJobId) {
        let mut span = None;
        let mut layout_of_depth = None;
        if let Some(map) = self.try_collect_active_jobs() {
            if let Some((info, depth)) = job.try_find_layout_root(map) {
                span = Some(info.job.span);
                layout_of_depth = Some(LayoutOfDepth { desc: info.query.description, depth });
            }
        }

        let suggested_limit = match self.recursion_limit() {
            Limit(0) => Limit(2),
            limit => limit * 2,
        };

        self.sess.emit_fatal(QueryOverflow {
            span,
            layout_of_depth,
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
    for query in &tcx.query_system.fns.query_structs {
        if let Some(encode) = query.encode_query_results {
            encode(tcx, encoder, query_result_index);
        }
    }
}

macro_rules! handle_cycle_error {
    ([]) => {{
        rustc_query_system::HandleCycleError::Error
    }};
    ([(fatal_cycle) $($rest:tt)*]) => {{
        rustc_query_system::HandleCycleError::Fatal
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
    ([][$qcx:expr, $name:ident, $key:expr]) => {{
        ($qcx.query_system.fns.local_providers.$name)($qcx, $key)
    }};
    ([(separate_provide_extern) $($rest:tt)*][$qcx:expr, $name:ident, $key:expr]) => {{
        if let Some(key) = $key.as_local_key() {
            ($qcx.query_system.fns.local_providers.$name)($qcx, key)
        } else {
            ($qcx.query_system.fns.extern_providers.$name)($qcx, $key)
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

pub(crate) fn create_query_frame<
    'tcx,
    K: Copy + Key + for<'a> HashStable<StableHashingContext<'a>>,
>(
    tcx: TyCtxt<'tcx>,
    do_describe: fn(TyCtxt<'tcx>, K) -> String,
    key: K,
    kind: DepKind,
    name: &'static str,
) -> QueryStackFrame<DepKind> {
    // Avoid calling queries while formatting the description
    let description = ty::print::with_no_queries!(
        // Disable visible paths printing for performance reasons.
        // Showing visible path instead of any path is not that important in production.
        ty::print::with_no_visible_paths!(
            // Force filename-line mode to avoid invoking `type_of` query.
            ty::print::with_forced_impl_filename_line!(do_describe(tcx, key))
        )
    );
    let description =
        if tcx.sess.verbose() { format!("{description} [{name:?}]") } else { description };
    let span = if kind == dep_graph::DepKind::def_span {
        // The `def_span` query is used to calculate `default_span`,
        // so exit to avoid infinite recursion.
        None
    } else {
        Some(key.default_span(tcx))
    };
    let def_id = key.key_as_def_id();
    let def_kind = if kind == dep_graph::DepKind::opt_def_kind {
        // Try to avoid infinite recursion.
        None
    } else {
        def_id.and_then(|def_id| def_id.as_local()).and_then(|def_id| tcx.opt_def_kind(def_id))
    };
    let hash = || {
        tcx.with_stable_hashing_context(|mut hcx| {
            let mut hasher = StableHasher::new();
            std::mem::discriminant(&kind).hash_stable(&mut hcx, &mut hasher);
            key.hash_stable(&mut hcx, &mut hasher);
            hasher.finish::<Hash64>()
        })
    };
    let ty_adt_id = key.ty_adt_id();

    QueryStackFrame::new(description, span, def_id, def_kind, kind, ty_adt_id, hash)
}

pub(crate) fn encode_query_results<'a, 'tcx, Q>(
    query: Q,
    qcx: QueryCtxt<'tcx>,
    encoder: &mut CacheEncoder<'a, 'tcx>,
    query_result_index: &mut EncodedDepNodeIndex,
) where
    Q: super::QueryConfigRestored<'tcx>,
    Q::RestoredValue: Encodable<CacheEncoder<'a, 'tcx>>,
{
    let _timer =
        qcx.profiler().verbose_generic_activity_with_arg("encode_query_results_for", query.name());

    assert!(query.query_state(qcx).all_inactive());
    let cache = query.query_cache(qcx);
    cache.iter(&mut |key, value, dep_node| {
        if query.cache_on_disk(qcx.tcx, &key) {
            let dep_node = SerializedDepNodeIndex::new(dep_node.index());

            // Record position of the cache entry.
            query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.position())));

            // Encode the type check tables with the `SerializedDepNodeIndex`
            // as tag.
            encoder.encode_tagged(dep_node, &Q::restore(*value));
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
    id: SerializedDepNodeIndex,
) -> Option<V>
where
    V: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
{
    tcx.query_system.on_disk_cache.as_ref()?.try_load_query_result(tcx, id)
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
        dep_node.kind != DepKind::codegen_unit,
        "calling force_from_dep_node() on DepKind::codegen_unit"
    );

    if let Some(key) = Q::Key::recover(tcx, &dep_node) {
        #[cfg(debug_assertions)]
        let _guard = tracing::span!(tracing::Level::TRACE, stringify!($name), ?key).entered();
        force_query(query, QueryCtxt::new(tcx), key, dep_node);
        true
    } else {
        false
    }
}

pub(crate) fn query_callback<'tcx, Q>(is_anon: bool, is_eval_always: bool) -> DepKindStruct<'tcx>
where
    Q: QueryConfig<QueryCtxt<'tcx>> + Default,
    Q::Key: DepNodeParams<TyCtxt<'tcx>>,
{
    let fingerprint_style = Q::Key::fingerprint_style();

    if is_anon || !fingerprint_style.reconstructible() {
        return DepKindStruct {
            is_anon,
            is_eval_always,
            fingerprint_style,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
        };
    }

    DepKindStruct {
        is_anon,
        is_eval_always,
        fingerprint_style,
        force_from_dep_node: Some(|tcx, dep_node| force_from_dep_node(Q::default(), tcx, dep_node)),
        try_load_from_on_disk_cache: Some(|tcx, dep_node| {
            try_load_from_on_disk_cache(Q::default(), tcx, dep_node)
        }),
    }
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
// invoked by `rustc_query_append`.
macro_rules! define_queries {
    (
     $($(#[$attr:meta])*
        [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {
        mod get_query {
            use super::*;

            $(
                #[inline(always)]
                #[tracing::instrument(level = "trace", skip(tcx))]
                pub(super) fn $name<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    span: Span,
                    key: query_keys::$name<'tcx>,
                    mode: QueryMode,
                ) -> Option<Erase<query_values::$name<'tcx>>> {
                    get_query(
                        queries::$name::default(),
                        QueryCtxt::new(tcx),
                        span,
                        key,
                        mode
                    )
                }
            )*
        }

        pub(crate) fn engine() -> QueryEngine {
            QueryEngine {
                $($name: get_query::$name,)*
            }
        }

        #[allow(nonstandard_style)]
        mod queries {
            use std::marker::PhantomData;

            $(
                #[derive(Copy, Clone, Default)]
                pub struct $name<'tcx> {
                    data: PhantomData<&'tcx ()>
                }
            )*
        }

        $(impl<'tcx> QueryConfig<QueryCtxt<'tcx>> for queries::$name<'tcx> {
            type Key = query_keys::$name<'tcx>;
            type Value = Erase<query_values::$name<'tcx>>;

            #[inline(always)]
            fn name(self) -> &'static str {
                stringify!($name)
            }

            #[inline]
            fn format_value(self) -> fn(&Self::Value) -> String {
                |value| format!("{:?}", restore::<query_values::$name<'tcx>>(*value))
            }

            #[inline]
            fn cache_on_disk(self, tcx: TyCtxt<'tcx>, key: &Self::Key) -> bool {
                ::rustc_middle::query::cached::$name(tcx, key)
            }

            type Cache = query_storage::$name<'tcx>;

            #[inline(always)]
            fn query_state<'a>(self, tcx: QueryCtxt<'tcx>) -> &'a QueryState<Self::Key, crate::dep_graph::DepKind>
                where QueryCtxt<'tcx>: 'a
            {
                &tcx.query_system.states.$name
            }

            #[inline(always)]
            fn query_cache<'a>(self, tcx: QueryCtxt<'tcx>) -> &'a Self::Cache
                where 'tcx:'a
            {
                &tcx.query_system.caches.$name
            }

            fn execute_query(self, tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
                erase(tcx.$name(key))
            }

            #[inline]
            #[allow(unused_variables)]
            fn compute(self, qcx: QueryCtxt<'tcx>, key: Self::Key) -> Self::Value {
                query_provided_to_value::$name(
                    qcx.tcx,
                    call_provider!([$($modifiers)*][qcx.tcx, $name, key])
                )
            }

            #[inline]
            fn try_load_from_disk(
                self,
                _qcx: QueryCtxt<'tcx>,
                _key: &Self::Key
            ) -> rustc_query_system::query::TryLoadFromDisk<QueryCtxt<'tcx>, Self::Value> {
                should_ever_cache_on_disk!([$($modifiers)*] {
                    if ::rustc_middle::query::cached::$name(_qcx.tcx, _key) {
                        Some(|qcx: QueryCtxt<'tcx>, dep_node| {
                            let value = $crate::plumbing::try_load_from_disk::<query_provided::$name<'tcx>>(
                                qcx.tcx,
                                dep_node
                            );
                            value.map(|value| query_provided_to_value::$name(qcx.tcx, value))
                        })
                    } else {
                        None
                    }
                } {
                    None
                })
            }

            #[inline]
            fn loadable_from_disk(
                self,
                _qcx: QueryCtxt<'tcx>,
                _key: &Self::Key,
                _index: SerializedDepNodeIndex,
            ) -> bool {
                should_ever_cache_on_disk!([$($modifiers)*] {
                    self.cache_on_disk(_qcx.tcx, _key) &&
                        $crate::plumbing::loadable_from_disk(_qcx.tcx, _index)
                } {
                    false
                })
            }

            #[inline]
            fn value_from_cycle_error(
                self,
                tcx: TyCtxt<'tcx>,
                cycle: &[QueryInfo<DepKind>],
            ) -> Self::Value {
                let result: query_values::$name<'tcx> = Value::from_cycle_error(tcx, cycle);
                erase(result)
            }

            #[inline(always)]
            fn anon(self) -> bool {
                is_anon!([$($modifiers)*])
            }

            #[inline(always)]
            fn eval_always(self) -> bool {
                is_eval_always!([$($modifiers)*])
            }

            #[inline(always)]
            fn depth_limit(self) -> bool {
                depth_limit!([$($modifiers)*])
            }

            #[inline(always)]
            fn feedable(self) -> bool {
                feedable!([$($modifiers)*])
            }

            #[inline(always)]
            fn dep_kind(self) -> rustc_middle::dep_graph::DepKind {
                dep_graph::DepKind::$name
            }

            #[inline(always)]
            fn handle_cycle_error(self) -> rustc_query_system::HandleCycleError {
                handle_cycle_error!([$($modifiers)*])
            }

            #[inline(always)]
            fn hash_result(self) -> rustc_query_system::query::HashResult<Self::Value> {
                hash_result!([$($modifiers)*][query_values::$name<'tcx>])
            }
        })*

        $(impl<'tcx> QueryConfigRestored<'tcx> for queries::$name<'tcx> {
            type RestoredValue = query_values::$name<'tcx>;

            #[inline(always)]
            fn restore(value: <Self as QueryConfig<QueryCtxt<'tcx>>>::Value) -> Self::RestoredValue {
                restore::<query_values::$name<'tcx>>(value)
            }
        })*

        #[allow(nonstandard_style)]
        mod query_callbacks {
            use super::*;
            use rustc_query_system::dep_graph::FingerprintStyle;

            // We use this for most things when incr. comp. is turned off.
            pub fn Null<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|_, dep_node| bug!("force_from_dep_node: encountered {:?}", dep_node)),
                    try_load_from_on_disk_cache: None,
                }
            }

            // We use this for the forever-red node.
            pub fn Red<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|_, dep_node| bug!("force_from_dep_node: encountered {:?}", dep_node)),
                    try_load_from_on_disk_cache: None,
                }
            }

            pub fn TraitSelect<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: true,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                }
            }

            pub fn CompileCodegenUnit<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                }
            }

            pub fn CompileMonoItem<'tcx>() -> DepKindStruct<'tcx> {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                }
            }

            $(pub(crate) fn $name<'tcx>()-> DepKindStruct<'tcx> {
                $crate::plumbing::query_callback::<queries::$name<'tcx>>(
                    is_anon!([$($modifiers)*]),
                    is_eval_always!([$($modifiers)*]),
                )
            })*
        }

        mod query_structs {
            use super::*;
            use rustc_middle::ty::query::QueryStruct;
            use rustc_middle::ty::query::QueryKeyStringCache;
            use rustc_middle::dep_graph::DepKind;

            pub(super) const fn dummy_query_struct<'tcx>() -> QueryStruct<'tcx> {
                fn noop_try_collect_active_jobs(_: TyCtxt<'_>, _: &mut QueryMap<DepKind>) -> Option<()> {
                    None
                }
                fn noop_alloc_self_profile_query_strings(_: TyCtxt<'_>, _: &mut QueryKeyStringCache) {}

                QueryStruct {
                    try_collect_active_jobs: noop_try_collect_active_jobs,
                    alloc_self_profile_query_strings: noop_alloc_self_profile_query_strings,
                    encode_query_results: None,
                }
            }

            pub(super) use dummy_query_struct as Null;
            pub(super) use dummy_query_struct as Red;
            pub(super) use dummy_query_struct as TraitSelect;
            pub(super) use dummy_query_struct as CompileCodegenUnit;
            pub(super) use dummy_query_struct as CompileMonoItem;

            $(
            pub(super) const fn $name<'tcx>() -> QueryStruct<'tcx> { QueryStruct {
                try_collect_active_jobs: |tcx, qmap| {
                    let make_query = |tcx, key| {
                        let kind = rustc_middle::dep_graph::DepKind::$name;
                        let name = stringify!($name);
                        $crate::plumbing::create_query_frame(tcx, rustc_middle::query::descs::$name, key, kind, name)
                    };
                    tcx.query_system.states.$name.try_collect_active_jobs(
                        tcx,
                        make_query,
                        qmap,
                    )
                },
                alloc_self_profile_query_strings: |tcx, string_cache| {
                    $crate::profiling_support::alloc_self_profile_query_strings_for_query_cache(
                        tcx,
                        stringify!($name),
                        &tcx.query_system.caches.$name,
                        string_cache,
                    )
                },
                encode_query_results: expand_if_cached!([$($modifiers)*], |tcx, encoder, query_result_index|
                    $crate::plumbing::encode_query_results::<super::queries::$name<'tcx>>(
                        super::queries::$name::default(),
                        QueryCtxt::new(tcx),
                        encoder,
                        query_result_index,
                    )
                ),
            }})*
        }

        pub fn query_callbacks<'tcx>(arena: &'tcx Arena<'tcx>) -> &'tcx [DepKindStruct<'tcx>] {
            arena.alloc_from_iter(make_dep_kind_array!(query_callbacks))
        }
    }
}
