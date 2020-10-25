//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::dep_graph::DepGraph;
use crate::ty::query::Query;
use crate::ty::tls::{self, ImplicitCtxt};
use crate::ty::{self, TyCtxt};
use rustc_query_system::query::QueryContext;
use rustc_query_system::query::{CycleError, QueryJobId, QueryJobInfo};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::{struct_span_err, Diagnostic, DiagnosticBuilder, Handler, Level};
use rustc_span::def_id::DefId;
use rustc_span::Span;

impl QueryContext for TyCtxt<'tcx> {
    type Query = Query<'tcx>;

    fn incremental_verify_ich(&self) -> bool {
        self.sess.opts.debugging_opts.incremental_verify_ich
    }
    fn verbose(&self) -> bool {
        self.sess.verbose()
    }

    fn def_path_str(&self, def_id: DefId) -> String {
        TyCtxt::def_path_str(*self, def_id)
    }

    fn dep_graph(&self) -> &DepGraph {
        &self.dep_graph
    }

    fn current_query_job(&self) -> Option<QueryJobId<Self::DepKind>> {
        tls::with_related_context(*self, |icx| icx.query)
    }

    fn try_collect_active_jobs(
        &self,
    ) -> Option<FxHashMap<QueryJobId<Self::DepKind>, QueryJobInfo<Self::DepKind, Self::Query>>>
    {
        self.queries.try_collect_active_jobs()
    }

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    #[inline(always)]
    fn start_query<R>(
        &self,
        token: QueryJobId<Self::DepKind>,
        diagnostics: Option<&Lock<ThinVec<Diagnostic>>>,
        compute: impl FnOnce(Self) -> R,
    ) -> R {
        // The `TyCtxt` stored in TLS has the same global interner lifetime
        // as `self`, so we use `with_related_context` to relate the 'tcx lifetimes
        // when accessing the `ImplicitCtxt`.
        tls::with_related_context(*self, move |current_icx| {
            // Update the `ImplicitCtxt` to point to our new query job.
            let new_icx = ImplicitCtxt {
                tcx: *self,
                query: Some(token),
                diagnostics,
                layout_depth: current_icx.layout_depth,
                task_deps: current_icx.task_deps,
            };

            // Use the `ImplicitCtxt` while we execute the query.
            tls::enter_context(&new_icx, |_| {
                rustc_data_structures::stack::ensure_sufficient_stack(|| compute(*self))
            })
        })
    }
}

impl<'tcx> TyCtxt<'tcx> {
    #[inline(never)]
    #[cold]
    pub(super) fn report_cycle(
        self,
        CycleError { usage, cycle: stack }: CycleError<Query<'tcx>>,
    ) -> DiagnosticBuilder<'tcx> {
        assert!(!stack.is_empty());

        let fix_span = |span: Span, query: &Query<'tcx>| {
            self.sess.source_map().guess_head_span(query.default_span(self, span))
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

    pub fn try_print_query_stack(handler: &Handler, num_frames: Option<usize>) {
        eprintln!("query stack during panic:");

        // Be careful reyling on global state here: this code is called from
        // a panic hook, which means that the global `Handler` may be in a weird
        // state if it was responsible for triggering the panic.
        let mut i = 0;
        ty::tls::with_context_opt(|icx| {
            if let Some(icx) = icx {
                let query_map = icx.tcx.queries.try_collect_active_jobs();

                let mut current_query = icx.query;

                while let Some(query) = current_query {
                    if Some(i) == num_frames {
                        break;
                    }
                    let query_info =
                        if let Some(info) = query_map.as_ref().and_then(|map| map.get(&query)) {
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
                            query_info.info.query.describe(icx.tcx)
                        ),
                    );
                    diag.span =
                        icx.tcx.sess.source_map().guess_head_span(query_info.info.span).into();
                    handler.force_print_diagnostic(diag);

                    current_query = query_info.job.parent;
                    i += 1;
                }
            }
        });

        if num_frames == None || num_frames >= Some(i) {
            eprintln!("end of query stack");
        } else {
            eprintln!("we're just showing a limited slice of the query stack");
        }
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
        <<$K as Key>::CacheSelector as CacheSelector<$K, $V>>::Cache
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

            pub fn describe(&self, tcx: TyCtxt<$tcx>) -> Cow<'static, str> {
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

        $(impl<$tcx> QueryConfig for queries::$name<$tcx> {
            type Key = $($K)*;
            type Value = $V;
            type Stored = <
                query_storage!([$($modifiers)*][$($K)*, $V])
                as QueryStorage
            >::Stored;
            const NAME: &'static str = stringify!($name);
        }

        impl<$tcx> QueryAccessors<TyCtxt<$tcx>> for queries::$name<$tcx> {
            const ANON: bool = is_anon!([$($modifiers)*]);
            const EVAL_ALWAYS: bool = is_eval_always!([$($modifiers)*]);
            const DEP_KIND: dep_graph::DepKind = dep_graph::DepKind::$name;

            type Cache = query_storage!([$($modifiers)*][$($K)*, $V]);

            #[inline(always)]
            fn query_state<'a>(tcx: TyCtxt<$tcx>) -> &'a QueryState<crate::dep_graph::DepKind, <TyCtxt<$tcx> as QueryContext>::Query, Self::Cache> {
                &tcx.queries.$name
            }

            #[inline]
            fn compute(tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value {
                let provider = tcx.queries.providers.get(key.query_crate())
                    // HACK(eddyb) it's possible crates may be loaded after
                    // the query engine is created, and because crate loading
                    // is not yet integrated with the query engine, such crates
                    // would be missing appropriate entries in `providers`.
                    .unwrap_or(&tcx.queries.fallback_extern_providers)
                    .$name;
                provider(tcx, key)
            }

            fn hash_result(
                _hcx: &mut StableHashingContext<'_>,
                _result: &Self::Value
            ) -> Option<Fingerprint> {
                hash_result!([$($modifiers)*][_hcx, _result])
            }

            fn handle_cycle_error(
                tcx: TyCtxt<'tcx>,
                error: CycleError<Query<'tcx>>
            ) -> Self::Value {
                handle_cycle_error!([$($modifiers)*][tcx, error])
            }
        })*

        #[derive(Copy, Clone)]
        pub struct TyCtxtEnsure<'tcx> {
            pub tcx: TyCtxt<'tcx>,
        }

        impl TyCtxtEnsure<$tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) {
                ensure_query::<queries::$name<'_>, _>(self.tcx, key.into_query_param())
            })*
        }

        #[derive(Copy, Clone)]
        pub struct TyCtxtAt<'tcx> {
            pub tcx: TyCtxt<'tcx>,
            pub span: Span,
        }

        impl Deref for TyCtxtAt<'tcx> {
            type Target = TyCtxt<'tcx>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.tcx
            }
        }

        impl TyCtxt<$tcx> {
            /// Returns a transparent wrapper for `TyCtxt`, which ensures queries
            /// are executed instead of just returning their results.
            #[inline(always)]
            pub fn ensure(self) -> TyCtxtEnsure<$tcx> {
                TyCtxtEnsure {
                    tcx: self,
                }
            }

            /// Returns a transparent wrapper for `TyCtxt` which uses
            /// `span` as the location of queries performed through it.
            #[inline(always)]
            pub fn at(self, span: Span) -> TyCtxtAt<$tcx> {
                TyCtxtAt {
                    tcx: self,
                    span
                }
            }

            $($(#[$attr])*
            #[inline(always)]
            #[must_use]
            pub fn $name(self, key: query_helper_param_ty!($($K)*))
                -> <queries::$name<$tcx> as QueryConfig>::Stored
            {
                self.at(DUMMY_SP).$name(key.into_query_param())
            })*

            /// All self-profiling events generated by the query engine use
            /// virtual `StringId`s for their `event_id`. This method makes all
            /// those virtual `StringId`s point to actual strings.
            ///
            /// If we are recording only summary data, the ids will point to
            /// just the query names. If we are recording query keys too, we
            /// allocate the corresponding strings here.
            pub fn alloc_self_profile_query_strings(self) {
                use crate::ty::query::profiling_support::{
                    alloc_self_profile_query_strings_for_query_cache,
                    QueryKeyStringCache,
                };

                if !self.prof.enabled() {
                    return;
                }

                let mut string_cache = QueryKeyStringCache::new();

                $({
                    alloc_self_profile_query_strings_for_query_cache(
                        self,
                        stringify!($name),
                        &self.queries.$name,
                        &mut string_cache,
                    );
                })*
            }
        }

        impl TyCtxtAt<$tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*))
                -> <queries::$name<$tcx> as QueryConfig>::Stored
            {
                get_query::<queries::$name<'_>, _>(self.tcx, self.span, key.into_query_param())
            })*
        }

        define_provider_struct! {
            tcx: $tcx,
            input: ($(([$($modifiers)*] [$name] [$($K)*] [$V]))*)
        }

        impl Copy for Providers {}
        impl Clone for Providers {
            fn clone(&self) -> Self { *self }
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
            /// This provides access to the incrimental comilation on-disk cache for query results.
            /// Do not access this directly. It is only meant to be used by
            /// `DepGraph::try_mark_green()` and the query infrastructure.
            pub(crate) on_disk_cache: OnDiskCache<'tcx>,

            providers: IndexVec<CrateNum, Providers>,
            fallback_extern_providers: Box<Providers>,

            $($(#[$attr])*  $name: QueryState<
                crate::dep_graph::DepKind,
                <TyCtxt<$tcx> as QueryContext>::Query,
                <queries::$name<$tcx> as QueryAccessors<TyCtxt<'tcx>>>::Cache,
            >,)*
        }

        impl<$tcx> Queries<$tcx> {
            pub(crate) fn new(
                providers: IndexVec<CrateNum, Providers>,
                fallback_extern_providers: Providers,
                on_disk_cache: OnDiskCache<'tcx>,
            ) -> Self {
                Queries {
                    providers,
                    fallback_extern_providers: Box::new(fallback_extern_providers),
                    on_disk_cache,
                    $($name: Default::default()),*
                }
            }

            pub(crate) fn try_collect_active_jobs(
                &self
            ) -> Option<FxHashMap<QueryJobId<crate::dep_graph::DepKind>, QueryJobInfo<crate::dep_graph::DepKind, <TyCtxt<$tcx> as QueryContext>::Query>>> {
                let mut jobs = FxHashMap::default();

                $(
                    self.$name.try_collect_active_jobs(
                        <queries::$name<'tcx> as QueryAccessors<TyCtxt<'tcx>>>::DEP_KIND,
                        Query::$name,
                        &mut jobs,
                    )?;
                )*

                Some(jobs)
            }
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
    };
}
