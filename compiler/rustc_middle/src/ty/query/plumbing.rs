//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph;
use crate::ty::query::{on_disk_cache, Queries, Query};
use crate::ty::tls::{self, ImplicitCtxt};
use crate::ty::{self, TyCtxt};
use rustc_query_system::dep_graph::HasDepContext;
use rustc_query_system::query::QueryContext;
use rustc_query_system::query::{CycleError, QueryJobId, QueryJobInfo};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::{struct_span_err, Diagnostic, DiagnosticBuilder, Handler, Level};
use rustc_serialize::opaque;
use rustc_span::def_id::DefId;
use rustc_span::Span;

#[derive(Copy, Clone)]
pub struct QueryCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub queries: &'tcx super::Queries<'tcx>,
}

impl<'tcx> std::ops::Deref for QueryCtxt<'tcx> {
    type Target = TyCtxt<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.tcx
    }
}

impl HasDepContext for QueryCtxt<'tcx> {
    type DepKind = crate::dep_graph::DepKind;
    type StableHashingContext = crate::ich::StableHashingContext<'tcx>;
    type DepContext = TyCtxt<'tcx>;

    #[inline]
    fn dep_context(&self) -> &Self::DepContext {
        &self.tcx
    }
}

impl QueryContext for QueryCtxt<'tcx> {
    type Query = Query<'tcx>;

    fn incremental_verify_ich(&self) -> bool {
        self.sess.opts.debugging_opts.incremental_verify_ich
    }
    fn verbose(&self) -> bool {
        self.sess.verbose()
    }

    fn def_path_str(&self, def_id: DefId) -> String {
        self.tcx.def_path_str(def_id)
    }

    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>> {
        tls::with_related_context(**self, |icx| icx.query)
    }

    fn try_collect_active_jobs(
        &self,
    ) -> Option<FxHashMap<QueryJobId<Self::DepKind>, QueryJobInfo<Self::DepKind, Self::Query>>>
    {
        self.queries.try_collect_active_jobs()
    }

    fn try_load_from_on_disk_cache(&self, dep_node: &dep_graph::DepNode) {
        (dep_node.kind.try_load_from_on_disk_cache)(*self, dep_node)
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
    #[inline(never)]
    #[cold]
    pub(super) fn report_cycle(
        self,
        CycleError { usage, cycle: stack }: CycleError<Query<'tcx>>,
    ) -> DiagnosticBuilder<'tcx> {
        assert!(!stack.is_empty());

        let fix_span = |span: Span, query: &Query<'tcx>| {
            self.sess.source_map().guess_head_span(query.default_span(*self, span))
        };

        // Disable naming impls with types in this path, since that
        // sometimes cycles itself, leading to extra cycle errors.
        // (And cycle errors around impls tend to occur during the
        // collect/coherence phases anyhow.)
        ty::print::with_forced_impl_filename_line(|| {
            let span = fix_span(stack[1 % stack.len()].span, &stack[0].query);
            let mut err = struct_span_err!(
                self.sess,
                span,
                E0391,
                "cycle detected when {}",
                stack[0].query.describe(self)
            );

            for i in 1..stack.len() {
                let query = &stack[i].query;
                let span = fix_span(stack[(i + 1) % stack.len()].span, query);
                err.span_note(span, &format!("...which requires {}...", query.describe(self)));
            }

            err.note(&format!(
                "...which again requires {}, completing the cycle",
                stack[0].query.describe(self)
            ));

            if let Some((span, query)) = usage {
                err.span_note(
                    fix_span(span, &query),
                    &format!("cycle used when {}", query.describe(self)),
                );
            }

            err
        })
    }

    pub(super) fn encode_query_results(
        self,
        encoder: &mut on_disk_cache::CacheEncoder<'a, 'tcx, opaque::FileEncoder>,
        query_result_index: &mut on_disk_cache::EncodedQueryResultIndex,
    ) -> opaque::FileEncodeResult {
        macro_rules! encode_queries {
            ($($query:ident,)*) => {
                $(
                    on_disk_cache::encode_query_results::<ty::query::queries::$query<'_>>(
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
}

impl<'tcx> Queries<'tcx> {
    pub fn try_print_query_stack(
        &'tcx self,
        tcx: TyCtxt<'tcx>,
        query: Option<QueryJobId<dep_graph::DepKind>>,
        handler: &Handler,
        num_frames: Option<usize>,
    ) -> usize {
        let query_map = self.try_collect_active_jobs();

        let mut current_query = query;
        let mut i = 0;

        while let Some(query) = current_query {
            if Some(i) == num_frames {
                break;
            }
            let query_info = if let Some(info) = query_map.as_ref().and_then(|map| map.get(&query))
            {
                info
            } else {
                break;
            };
            let mut diag = Diagnostic::new(
                Level::FailureNote,
                &format!(
                    "#{} [{}] {}",
                    i,
                    query_info.info.query.name(),
                    query_info.info.query.describe(QueryCtxt { tcx, queries: self })
                ),
            );
            diag.span = tcx.sess.source_map().guess_head_span(query_info.info.span).into();
            handler.force_print_diagnostic(diag);

            current_query = query_info.job.parent;
            i += 1;
        }

        i
    }
}

macro_rules! handle_cycle_error {
    ([][$tcx: expr, $error:expr]) => {{
        $tcx.report_cycle($error).emit();
        Value::from_cycle_error($tcx)
    }};
    ([fatal_cycle $($rest:tt)*][$tcx:expr, $error:expr]) => {{
        $tcx.report_cycle($error).emit();
        $tcx.sess.abort_if_errors();
        unreachable!()
    }};
    ([cycle_delay_bug $($rest:tt)*][$tcx:expr, $error:expr]) => {{
        $tcx.report_cycle($error).delay_as_bug();
        Value::from_cycle_error($tcx)
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*][$($args:tt)*]) => {
        handle_cycle_error!([$($($modifiers)*)*][$($args)*])
    };
}

macro_rules! is_anon {
    ([]) => {{
        false
    }};
    ([anon $($rest:tt)*]) => {{
        true
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*]) => {
        is_anon!([$($($modifiers)*)*])
    };
}

macro_rules! is_eval_always {
    ([]) => {{
        false
    }};
    ([eval_always $($rest:tt)*]) => {{
        true
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*]) => {
        is_eval_always!([$($($modifiers)*)*])
    };
}

macro_rules! query_storage {
    ([][$K:ty, $V:ty]) => {
        <DefaultCacheSelector as CacheSelector<$K, $V>>::Cache
    };
    ([storage($ty:ty) $($rest:tt)*][$K:ty, $V:ty]) => {
        <$ty as CacheSelector<$K, $V>>::Cache
    };
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*][$($args:tt)*]) => {
        query_storage!([$($($modifiers)*)*][$($args)*])
    };
}

macro_rules! hash_result {
    ([][$hcx:expr, $result:expr]) => {{
        dep_graph::hash_result($hcx, &$result)
    }};
    ([no_hash $($rest:tt)*][$hcx:expr, $result:expr]) => {{
        None
    }};
    ([$other:ident $(($($other_args:tt)*))* $(, $($modifiers:tt)*)*][$($args:tt)*]) => {
        hash_result!([$($($modifiers)*)*][$($args)*])
    };
}

macro_rules! query_helper_param_ty {
    (DefId) => { impl IntoQueryParam<DefId> };
    ($K:ty) => { $K };
}

macro_rules! define_queries {
    (<$tcx:tt>
     $($(#[$attr:meta])*
        [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {

        use std::mem;
        use crate::{
            rustc_data_structures::stable_hasher::HashStable,
            rustc_data_structures::stable_hasher::StableHasher,
            ich::StableHashingContext
        };

        define_queries_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$($attr)*] [$name]))*)
        }

        #[allow(nonstandard_style)]
        #[derive(Clone, Debug)]
        pub enum Query<$tcx> {
            $($(#[$attr])* $name($($K)*)),*
        }

        impl<$tcx> Query<$tcx> {
            pub fn name(&self) -> &'static str {
                match *self {
                    $(Query::$name(_) => stringify!($name),)*
                }
            }

            pub(crate) fn describe(&self, tcx: QueryCtxt<$tcx>) -> Cow<'static, str> {
                let (r, name) = match *self {
                    $(Query::$name(key) => {
                        (queries::$name::describe(tcx, key), stringify!($name))
                    })*
                };
                if tcx.sess.verbose() {
                    format!("{} [{}]", r, name).into()
                } else {
                    r
                }
            }

            // FIXME(eddyb) Get more valid `Span`s on queries.
            pub fn default_span(&self, tcx: TyCtxt<$tcx>, span: Span) -> Span {
                if !span.is_dummy() {
                    return span;
                }
                // The `def_span` query is used to calculate `default_span`,
                // so exit to avoid infinite recursion.
                if let Query::def_span(..) = *self {
                    return span
                }
                match *self {
                    $(Query::$name(key) => key.default_span(tcx),)*
                }
            }
        }

        impl<'a, $tcx> HashStable<StableHashingContext<'a>> for Query<$tcx> {
            fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
                mem::discriminant(self).hash_stable(hcx, hasher);
                match *self {
                    $(Query::$name(key) => key.hash_stable(hcx, hasher),)*
                }
            }
        }

        #[allow(nonstandard_style)]
        pub mod queries {
            use std::marker::PhantomData;

            $(pub struct $name<$tcx> {
                data: PhantomData<&$tcx ()>
            })*
        }

        // HACK(eddyb) this is like the `impl QueryConfig for queries::$name`
        // below, but using type aliases instead of associated types, to bypass
        // the limitations around normalizing under HRTB - for example, this:
        // `for<'tcx> fn(...) -> <queries::$name<'tcx> as QueryConfig<TyCtxt<'tcx>>>::Value`
        // doesn't currently normalize to `for<'tcx> fn(...) -> query_values::$name<'tcx>`.
        // This is primarily used by the `provide!` macro in `rustc_metadata`.
        #[allow(nonstandard_style, unused_lifetimes)]
        pub mod query_keys {
            use super::*;

            $(pub type $name<$tcx> = $($K)*;)*
        }
        #[allow(nonstandard_style, unused_lifetimes)]
        pub mod query_values {
            use super::*;

            $(pub type $name<$tcx> = $V;)*
        }
        #[allow(nonstandard_style, unused_lifetimes)]
        pub mod query_stored {
            use super::*;

            $(pub type $name<$tcx> = <
                query_storage!([$($modifiers)*][$($K)*, $V])
                as QueryStorage
            >::Stored;)*
        }

        $(impl<$tcx> QueryConfig for queries::$name<$tcx> {
            type Key = $($K)*;
            type Value = $V;
            type Stored = query_stored::$name<$tcx>;
            const NAME: &'static str = stringify!($name);
        }

        impl<$tcx> QueryAccessors<QueryCtxt<$tcx>> for queries::$name<$tcx> {
            const ANON: bool = is_anon!([$($modifiers)*]);
            const EVAL_ALWAYS: bool = is_eval_always!([$($modifiers)*]);
            const DEP_KIND: dep_graph::DepKind = dep_graph::DepKind::$name;

            type Cache = query_storage!([$($modifiers)*][$($K)*, $V]);

            #[inline(always)]
            fn query_state<'a>(tcx: QueryCtxt<$tcx>) -> &'a QueryState<crate::dep_graph::DepKind, Query<$tcx>, Self::Cache> {
                &tcx.queries.$name
            }

            #[inline]
            fn compute(tcx: QueryCtxt<'tcx>, key: Self::Key) -> Self::Value {
                let provider = tcx.queries.providers.get(key.query_crate())
                    // HACK(eddyb) it's possible crates may be loaded after
                    // the query engine is created, and because crate loading
                    // is not yet integrated with the query engine, such crates
                    // would be missing appropriate entries in `providers`.
                    .unwrap_or(&tcx.queries.fallback_extern_providers)
                    .$name;
                provider(*tcx, key)
            }

            fn hash_result(
                _hcx: &mut StableHashingContext<'_>,
                _result: &Self::Value
            ) -> Option<Fingerprint> {
                hash_result!([$($modifiers)*][_hcx, _result])
            }

            fn handle_cycle_error(
                tcx: QueryCtxt<'tcx>,
                error: CycleError<Query<'tcx>>
            ) -> Self::Value {
                handle_cycle_error!([$($modifiers)*][tcx, error])
            }
        })*

        define_provider_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$name] [$($K)*] [$V]))*)
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
            providers: IndexVec<CrateNum, Providers>,
            fallback_extern_providers: Box<Providers>,

            $($(#[$attr])*  $name: QueryState<
                crate::dep_graph::DepKind,
                Query<$tcx>,
                <queries::$name<$tcx> as QueryAccessors<QueryCtxt<'tcx>>>::Cache,
            >,)*
        }

        impl<$tcx> Queries<$tcx> {
            pub fn new(
                providers: IndexVec<CrateNum, Providers>,
                fallback_extern_providers: Providers,
            ) -> Self {
                Queries {
                    providers,
                    fallback_extern_providers: Box::new(fallback_extern_providers),
                    $($name: Default::default()),*
                }
            }

            /// All self-profiling events generated by the query engine use
            /// virtual `StringId`s for their `event_id`. This method makes all
            /// those virtual `StringId`s point to actual strings.
            ///
            /// If we are recording only summary data, the ids will point to
            /// just the query names. If we are recording query keys too, we
            /// allocate the corresponding strings here.
            pub fn alloc_self_profile_query_strings(&self, tcx: TyCtxt<'tcx>) {
                use crate::ty::query::profiling_support::{
                    alloc_self_profile_query_strings_for_query_cache,
                    QueryKeyStringCache,
                };

                if !tcx.prof.enabled() {
                    return;
                }

                let mut string_cache = QueryKeyStringCache::new();

                $({
                    alloc_self_profile_query_strings_for_query_cache(
                        tcx,
                        stringify!($name),
                        &self.$name,
                        &mut string_cache,
                    );
                })*
            }

            pub(crate) fn try_collect_active_jobs(
                &self
            ) -> Option<FxHashMap<QueryJobId<crate::dep_graph::DepKind>, QueryJobInfo<crate::dep_graph::DepKind, Query<$tcx>>>> {
                let mut jobs = FxHashMap::default();

                $(
                    self.$name.try_collect_active_jobs(
                        <queries::$name<'tcx> as QueryAccessors<QueryCtxt<'tcx>>>::DEP_KIND,
                        Query::$name,
                        &mut jobs,
                    )?;
                )*

                Some(jobs)
            }

            #[cfg(parallel_compiler)]
            pub unsafe fn deadlock(&'tcx self, tcx: TyCtxt<'tcx>, registry: &rustc_rayon_core::Registry) {
                let tcx = QueryCtxt { tcx, queries: self };
                rustc_query_system::query::deadlock(tcx, registry)
            }

            pub(crate) fn encode_query_results(
                &'tcx self,
                tcx: TyCtxt<'tcx>,
                encoder: &mut on_disk_cache::CacheEncoder<'a, 'tcx, opaque::FileEncoder>,
                query_result_index: &mut on_disk_cache::EncodedQueryResultIndex,
            ) -> opaque::FileEncodeResult {
                let tcx = QueryCtxt { tcx, queries: self };
                tcx.encode_query_results(encoder, query_result_index)
            }

            fn exec_cache_promotions(&'tcx self, tcx: TyCtxt<'tcx>) {
                let tcx = QueryCtxt { tcx, queries: self };
                tcx.dep_graph.exec_cache_promotions(tcx)
            }

            $($(#[$attr])*
            #[inline(always)]
            fn $name(
                &'tcx self,
                tcx: TyCtxt<$tcx>,
                span: Span,
                key: query_keys::$name<$tcx>,
                mode: QueryMode,
            ) -> Option<query_stored::$name<$tcx>> {
                let qcx = QueryCtxt { tcx, queries: self };
                get_query::<queries::$name<$tcx>, _>(qcx, span, key, mode)
            })*
        }
    };
}

macro_rules! define_provider_struct {
    (tcx: $tcx:tt,
     input: ($(([$($modifiers:tt)*] [$name:ident] [$K:ty] [$R:ty]))*)) => {
        pub struct Providers {
            $(pub $name: for<$tcx> fn(TyCtxt<$tcx>, $K) -> $R,)*
        }

        impl Default for Providers {
            fn default() -> Self {
                $(fn $name<$tcx>(_: TyCtxt<$tcx>, key: $K) -> $R {
                    bug!("`tcx.{}({:?})` unsupported by its crate",
                         stringify!($name), key);
                })*
                Providers { $($name),* }
            }
        }

        impl Copy for Providers {}
        impl Clone for Providers {
            fn clone(&self) -> Self { *self }
        }
    };
}
