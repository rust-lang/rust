//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::{on_disk_cache, queries, Queries};
use rustc_middle::dep_graph::{DepKind, DepNodeIndex, SerializedDepNodeIndex};
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::ty::{self, TyCtxt};
use rustc_query_system::dep_graph::HasDepContext;
use rustc_query_system::query::{
    QueryContext, QueryDescription, QueryJobId, QueryMap, QuerySideEffects,
};

use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::{Diagnostic, Handler};
use rustc_serialize::opaque;
use rustc_span::def_id::LocalDefId;

use std::any::Any;

#[derive(Copy, Clone)]
pub struct QueryCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub queries: &'tcx Queries<'tcx>,
}

impl<'tcx> std::ops::Deref for QueryCtxt<'tcx> {
    type Target = TyCtxt<'tcx>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.tcx
    }
}

impl HasDepContext for QueryCtxt<'tcx> {
    type DepKind = rustc_middle::dep_graph::DepKind;
    type DepContext = TyCtxt<'tcx>;

    #[inline]
    fn dep_context(&self) -> &Self::DepContext {
        &self.tcx
    }
}

impl QueryContext for QueryCtxt<'tcx> {
    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>> {
        tls::with_related_context(**self, |icx| icx.query)
    }

    fn try_collect_active_jobs(&self) -> Option<QueryMap<Self::DepKind>> {
        self.queries.try_collect_active_jobs(**self)
    }

    // Interactions with on_disk_cache
    fn load_side_effects(&self, prev_dep_node_index: SerializedDepNodeIndex) -> QuerySideEffects {
        self.queries
            .on_disk_cache
            .as_ref()
            .map(|c| c.load_side_effects(**self, prev_dep_node_index))
            .unwrap_or_default()
    }

    fn store_side_effects(&self, dep_node_index: DepNodeIndex, side_effects: QuerySideEffects) {
        if let Some(c) = self.queries.on_disk_cache.as_ref() {
            c.store_side_effects(dep_node_index, side_effects)
        }
    }

    fn store_side_effects_for_anon_node(
        &self,
        dep_node_index: DepNodeIndex,
        side_effects: QuerySideEffects,
    ) {
        if let Some(c) = self.queries.on_disk_cache.as_ref() {
            c.store_side_effects_for_anon_node(dep_node_index, side_effects)
        }
    }

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    #[inline(always)]
    fn start_query<R>(
        &self,
        token: QueryJobId<Self::DepKind>,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
        compute: impl FnOnce() -> R,
    ) -> R {
        // The `TyCtxt` stored in TLS has the same global interner lifetime
        // as `self`, so we use `with_related_context` to relate the 'tcx lifetimes
        // when accessing the `ImplicitCtxt`.
        tls::with_related_context(**self, move |current_icx| {
            // Update the `ImplicitCtxt` to point to our new query job.
            let new_icx = ImplicitCtxt {
                tcx: **self,
                query: Some(token),
                diagnostics,
                layout_depth: current_icx.layout_depth,
                task_deps: current_icx.task_deps,
            };

            // Use the `ImplicitCtxt` while we execute the query.
            tls::enter_context(&new_icx, |_| {
                rustc_data_structures::stack::ensure_sufficient_stack(compute)
            })
        })
    }
}

impl<'tcx> QueryCtxt<'tcx> {
    #[inline]
    pub fn from_tcx(tcx: TyCtxt<'tcx>) -> Self {
        let queries = tcx.queries.as_any();
        let queries = unsafe {
            let queries = std::mem::transmute::<&dyn Any, &dyn Any>(queries);
            let queries = queries.downcast_ref().unwrap();
            let queries = std::mem::transmute::<&Queries<'_>, &Queries<'_>>(queries);
            queries
        };
        QueryCtxt { tcx, queries }
    }

    crate fn on_disk_cache(self) -> Option<&'tcx on_disk_cache::OnDiskCache<'tcx>> {
        self.queries.on_disk_cache.as_ref()
    }

    #[cfg(parallel_compiler)]
    pub unsafe fn deadlock(self, registry: &rustc_rayon_core::Registry) {
        rustc_query_system::query::deadlock(self, registry)
    }

    pub(super) fn encode_query_results(
        self,
        encoder: &mut on_disk_cache::CacheEncoder<'a, 'tcx, opaque::FileEncoder>,
        query_result_index: &mut on_disk_cache::EncodedDepNodeIndex,
    ) -> opaque::FileEncodeResult {
        macro_rules! encode_queries {
            ($($query:ident,)*) => {
                $(
                    on_disk_cache::encode_query_results::<_, super::queries::$query<'_>>(
                        self,
                        encoder,
                        query_result_index
                    )?;
                )*
            }
        }

        rustc_cached_queries!(encode_queries!);

        Ok(())
    }

    pub fn try_print_query_stack(
        self,
        query: Option<QueryJobId<DepKind>>,
        handler: &Handler,
        num_frames: Option<usize>,
    ) -> usize {
        rustc_query_system::query::print_query_stack(self, query, handler, num_frames)
    }
}

macro_rules! handle_cycle_error {
    ([][$tcx: expr, $error:expr]) => {{
        $error.emit();
        Value::from_cycle_error($tcx)
    }};
    ([(fatal_cycle) $($rest:tt)*][$tcx:expr, $error:expr]) => {{
        $error.emit();
        $tcx.sess.abort_if_errors();
        unreachable!()
    }};
    ([(cycle_delay_bug) $($rest:tt)*][$tcx:expr, $error:expr]) => {{
        $error.delay_as_bug();
        Value::from_cycle_error($tcx)
    }};
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        handle_cycle_error!([$($modifiers)*][$($args)*])
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

macro_rules! hash_result {
    ([]) => {{
        Some(dep_graph::hash_result)
    }};
    ([(no_hash) $($rest:tt)*]) => {{
        None
    }};
    ([$other:tt $($modifiers:tt)*]) => {
        hash_result!([$($modifiers)*])
    };
}

macro_rules! define_queries {
    (<$tcx:tt>
     $($(#[$attr:meta])*
        [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {

        define_queries_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$($attr)*] [$name]))*)
        }

        mod make_query {
            use super::*;

            // Create an eponymous constructor for each query.
            $(#[allow(nonstandard_style)] $(#[$attr])*
            pub fn $name<$tcx>(tcx: QueryCtxt<$tcx>, key: query_keys::$name<$tcx>) -> QueryStackFrame {
                let kind = dep_graph::DepKind::$name;
                let name = stringify!($name);
                // Disable visible paths printing for performance reasons.
                // Showing visible path instead of any path is not that important in production.
                let description = ty::print::with_no_visible_paths(
                    || ty::print::with_forced_impl_filename_line(
                    // Force filename-line mode to avoid invoking `type_of` query.
                    || queries::$name::describe(tcx, key)
                ));
                let description = if tcx.sess.verbose() {
                    format!("{} [{}]", description, name)
                } else {
                    description
                };
                let span = if kind == dep_graph::DepKind::def_span {
                    // The `def_span` query is used to calculate `default_span`,
                    // so exit to avoid infinite recursion.
                    None
                } else {
                    Some(key.default_span(*tcx))
                };
                let def_id = key.key_as_def_id();
                let def_kind = def_id
                    .and_then(|def_id| def_id.as_local())
                    // Use `tcx.hir().opt_def_kind()` to reduce the chance of
                    // accidentally triggering an infinite query loop.
                    .and_then(|def_id| tcx.hir().opt_def_kind(def_id))
                    .map(|def_kind| $crate::util::def_kind_to_simple_def_kind(def_kind));
                let hash = || {
                    let mut hcx = tcx.create_stable_hashing_context();
                    let mut hasher = StableHasher::new();
                    std::mem::discriminant(&kind).hash_stable(&mut hcx, &mut hasher);
                    key.hash_stable(&mut hcx, &mut hasher);
                    hasher.finish::<u64>()
                };

                QueryStackFrame::new(name, description, span, def_kind, hash)
            })*
        }

        #[allow(nonstandard_style)]
        pub mod queries {
            use std::marker::PhantomData;

            $(pub struct $name<$tcx> {
                data: PhantomData<&$tcx ()>
            })*
        }

        $(impl<$tcx> QueryConfig for queries::$name<$tcx> {
            type Key = query_keys::$name<$tcx>;
            type Value = query_values::$name<$tcx>;
            type Stored = query_stored::$name<$tcx>;
            const NAME: &'static str = stringify!($name);
        }

        impl<$tcx> QueryAccessors<QueryCtxt<$tcx>> for queries::$name<$tcx> {
            const ANON: bool = is_anon!([$($modifiers)*]);
            const EVAL_ALWAYS: bool = is_eval_always!([$($modifiers)*]);
            const DEP_KIND: dep_graph::DepKind = dep_graph::DepKind::$name;
            const HASH_RESULT: Option<fn(&mut StableHashingContext<'_>, &Self::Value) -> Fingerprint> = hash_result!([$($modifiers)*]);

            type Cache = query_storage::$name<$tcx>;

            #[inline(always)]
            fn query_state<'a>(tcx: QueryCtxt<$tcx>) -> &'a QueryState<crate::dep_graph::DepKind, Self::Key>
                where QueryCtxt<$tcx>: 'a
            {
                &tcx.queries.$name
            }

            #[inline(always)]
            fn query_cache<'a>(tcx: QueryCtxt<$tcx>) -> &'a QueryCacheStore<Self::Cache>
                where 'tcx:'a
            {
                &tcx.query_caches.$name
            }

            #[inline]
            fn compute_fn(tcx: QueryCtxt<'tcx>, key: &Self::Key) ->
                fn(TyCtxt<'tcx>, Self::Key) -> Self::Value
            {
                if key.query_crate_is_local() {
                    tcx.queries.local_providers.$name
                } else {
                    tcx.queries.extern_providers.$name
                }
            }

            fn handle_cycle_error(
                tcx: QueryCtxt<'tcx>,
                mut error: DiagnosticBuilder<'_>,
            ) -> Self::Value {
                handle_cycle_error!([$($modifiers)*][tcx, error])
            }
        })*

        #[allow(nonstandard_style)]
        pub mod query_callbacks {
            use super::*;
            use rustc_middle::dep_graph::DepNode;
            use rustc_middle::ty::query::query_keys;
            use rustc_query_system::dep_graph::DepNodeParams;
            use rustc_query_system::query::{force_query, QueryDescription};
            use rustc_query_system::dep_graph::FingerprintStyle;

            // We use this for most things when incr. comp. is turned off.
            pub fn Null() -> DepKindStruct {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: Some(|_, dep_node| bug!("force_from_dep_node: encountered {:?}", dep_node)),
                    try_load_from_on_disk_cache: None,
                }
            }

            pub fn TraitSelect() -> DepKindStruct {
                DepKindStruct {
                    is_anon: true,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Unit,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                }
            }

            pub fn CompileCodegenUnit() -> DepKindStruct {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                }
            }

            pub fn CompileMonoItem() -> DepKindStruct {
                DepKindStruct {
                    is_anon: false,
                    is_eval_always: false,
                    fingerprint_style: FingerprintStyle::Opaque,
                    force_from_dep_node: None,
                    try_load_from_on_disk_cache: None,
                }
            }

            $(pub fn $name()-> DepKindStruct {
                let is_anon = is_anon!([$($modifiers)*]);
                let is_eval_always = is_eval_always!([$($modifiers)*]);

                let fingerprint_style =
                    <query_keys::$name<'_> as DepNodeParams<TyCtxt<'_>>>::fingerprint_style();

                if is_anon || !fingerprint_style.reconstructible() {
                    return DepKindStruct {
                        is_anon,
                        is_eval_always,
                        fingerprint_style,
                        force_from_dep_node: None,
                        try_load_from_on_disk_cache: None,
                    }
                }

                #[inline(always)]
                fn recover<'tcx>(tcx: TyCtxt<'tcx>, dep_node: DepNode) -> Option<query_keys::$name<'tcx>> {
                    <query_keys::$name<'_> as DepNodeParams<TyCtxt<'_>>>::recover(tcx, &dep_node)
                }

                fn force_from_dep_node(tcx: TyCtxt<'_>, dep_node: DepNode) -> bool {
                    if let Some(key) = recover(tcx, dep_node) {
                        let tcx = QueryCtxt::from_tcx(tcx);
                        force_query::<queries::$name<'_>, _>(tcx, key, dep_node);
                        true
                    } else {
                        false
                    }
                }

                fn try_load_from_on_disk_cache(tcx: TyCtxt<'_>, dep_node: DepNode) {
                    debug_assert!(tcx.dep_graph.is_green(&dep_node));

                    let key = recover(tcx, dep_node).unwrap_or_else(|| panic!("Failed to recover key for {:?} with hash {}", dep_node, dep_node.hash));
                    let tcx = QueryCtxt::from_tcx(tcx);
                    if queries::$name::cache_on_disk(tcx, &key, None) {
                        let _ = tcx.$name(key);
                    }
                }

                DepKindStruct {
                    is_anon,
                    is_eval_always,
                    fingerprint_style,
                    force_from_dep_node: Some(force_from_dep_node),
                    try_load_from_on_disk_cache: Some(try_load_from_on_disk_cache),
                }
            })*
        }

        pub fn query_callbacks<'tcx>(arena: &'tcx Arena<'tcx>) -> &'tcx [DepKindStruct] {
            arena.alloc_from_iter(make_dep_kind_array!(query_callbacks))
        }
    }
}

// FIXME(eddyb) this macro (and others?) use `$tcx` and `'tcx` interchangeably.
// We should either not take `$tcx` at all and use `'tcx` everywhere, or use
// `$tcx` everywhere (even if that isn't necessary due to lack of hygiene).
macro_rules! define_queries_struct {
    (tcx: $tcx:tt,
     input: ($(([$($modifiers:tt)*] [$($attr:tt)*] [$name:ident]))*)) => {
        pub struct Queries<$tcx> {
            local_providers: Box<Providers>,
            extern_providers: Box<Providers>,

            pub on_disk_cache: Option<OnDiskCache<$tcx>>,

            $($(#[$attr])*  $name: QueryState<
                crate::dep_graph::DepKind,
                query_keys::$name<$tcx>,
            >,)*
        }

        impl<$tcx> Queries<$tcx> {
            pub fn new(
                local_providers: Providers,
                extern_providers: Providers,
                on_disk_cache: Option<OnDiskCache<$tcx>>,
            ) -> Self {
                Queries {
                    local_providers: Box::new(local_providers),
                    extern_providers: Box::new(extern_providers),
                    on_disk_cache,
                    $($name: Default::default()),*
                }
            }

            pub(crate) fn try_collect_active_jobs(
                &$tcx self,
                tcx: TyCtxt<$tcx>,
            ) -> Option<QueryMap<crate::dep_graph::DepKind>> {
                let tcx = QueryCtxt { tcx, queries: self };
                let mut jobs = QueryMap::default();

                $(
                    self.$name.try_collect_active_jobs(
                        tcx,
                        dep_graph::DepKind::$name,
                        make_query::$name,
                        &mut jobs,
                    )?;
                )*

                Some(jobs)
            }
        }

        impl QueryEngine<'tcx> for Queries<'tcx> {
            fn as_any(&'tcx self) -> &'tcx dyn std::any::Any {
                let this = unsafe { std::mem::transmute::<&Queries<'_>, &Queries<'_>>(self) };
                this as _
            }

            fn try_mark_green(&'tcx self, tcx: TyCtxt<'tcx>, dep_node: &dep_graph::DepNode) -> bool {
                let qcx = QueryCtxt { tcx, queries: self };
                tcx.dep_graph.try_mark_green(qcx, dep_node).is_some()
            }

            $($(#[$attr])*
            #[inline(always)]
            fn $name(
                &'tcx self,
                tcx: TyCtxt<$tcx>,
                span: Span,
                key: query_keys::$name<$tcx>,
                lookup: QueryLookup,
                mode: QueryMode,
            ) -> Option<query_stored::$name<$tcx>> {
                let qcx = QueryCtxt { tcx, queries: self };
                get_query::<queries::$name<$tcx>, _>(qcx, span, key, lookup, mode)
            })*
        }
    };
}

fn describe_as_module(def_id: LocalDefId, tcx: TyCtxt<'_>) -> String {
    if def_id.is_top_level_module() {
        "top-level module".to_string()
    } else {
        format!("module `{}`", tcx.def_path_str(def_id.to_def_id()))
    }
}

rustc_query_description! {}
