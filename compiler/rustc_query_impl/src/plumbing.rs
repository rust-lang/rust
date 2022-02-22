//! The implementation of the query system itself. This defines the macros that
//! generate the actual methods on tcx which find and execute the provider,
//! manage the caches, and so forth.

use crate::Queries;
use rustc_middle::ty::tls::{self, ImplicitCtxt};
use rustc_middle::ty::TyCtxt;
use rustc_query_system::dep_graph::HasDepContext;
use rustc_query_system::query::{QueryContext, QueryJobId, QueryMap};

use rustc_errors::Handler;

use std::any::Any;
use std::num::NonZeroU64;

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

impl<'tcx> HasDepContext for QueryCtxt<'tcx> {
    type DepContext = TyCtxt<'tcx>;

    #[inline]
    fn dep_context(&self) -> &Self::DepContext {
        &self.tcx
    }
}

impl QueryContext for QueryCtxt<'_> {
    fn next_job_id(&self) -> QueryJobId {
        QueryJobId(
            NonZeroU64::new(
                self.queries.jobs.fetch_add(1, rustc_data_structures::sync::Ordering::Relaxed),
            )
            .unwrap(),
        )
    }

    fn current_query_job(&self) -> Option<QueryJobId> {
        tls::with_related_context(**self, |icx| icx.query)
    }

    fn try_collect_active_jobs(&self) -> Option<QueryMap> {
        self.queries.try_collect_active_jobs(**self)
    }

    /// Executes a job by changing the `ImplicitCtxt` to point to the
    /// new query job while it executes. It returns the diagnostics
    /// captured during execution and the actual result.
    #[inline(always)]
    fn start_query<R>(&self, token: QueryJobId, compute: impl FnOnce() -> R) -> R {
        // The `TyCtxt` stored in TLS has the same global interner lifetime
        // as `self`, so we use `with_related_context` to relate the 'tcx lifetimes
        // when accessing the `ImplicitCtxt`.
        tls::with_related_context(**self, move |current_icx| {
            // Update the `ImplicitCtxt` to point to our new query job.
            let new_icx = ImplicitCtxt {
                tcx: **self,
                query: Some(token),
                layout_depth: current_icx.layout_depth,
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

    #[cfg(parallel_compiler)]
    pub unsafe fn deadlock(self, registry: &rustc_rayon_core::Registry) {
        rustc_query_system::query::deadlock(self, registry)
    }

    pub fn try_print_query_stack(
        self,
        query: Option<QueryJobId>,
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

macro_rules! get_provider {
    ([][$tcx:expr, $name:ident, $key:expr]) => {{
        $tcx.queries.local_providers.$name
    }};
    ([(separate_provide_extern) $($rest:tt)*][$tcx:expr, $name:ident, $key:expr]) => {{
        if $key.query_crate_is_local() {
            $tcx.queries.local_providers.$name
        } else {
            $tcx.queries.extern_providers.$name
        }
    }};
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        get_provider!([$($modifiers)*][$($args)*])
    };
}

macro_rules! opt_remap_env_constness {
    ([][$name:ident]) => {};
    ([(remap_env_constness) $($rest:tt)*][$name:ident]) => {
        let $name = $name.without_const();
    };
    ([$other:tt $($modifiers:tt)*][$name:ident]) => {
        opt_remap_env_constness!([$($modifiers)*][$name])
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
                opt_remap_env_constness!([$($modifiers)*][key]);
                let kind = dep_graph::DepKind::$name;
                let name = stringify!($name);
                // Disable visible paths printing for performance reasons.
                // Showing visible path instead of any path is not that important in production.
                let description = ty::print::with_no_visible_paths!(
                    // Force filename-line mode to avoid invoking `type_of` query.
                    ty::print::with_forced_impl_filename_line!(
                        queries::$name::describe(tcx, key)
                    )
                );
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
                // Use `tcx.hir().opt_def_kind()` to reduce the chance of
                // accidentally triggering an infinite query loop.
                let def_kind = key.key_as_def_id()
                    .and_then(|def_id| def_id.as_local())
                    .and_then(|def_id| tcx.hir().opt_def_kind(def_id));
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
        mod queries {
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

        impl<$tcx> QueryDescription<QueryCtxt<$tcx>> for queries::$name<$tcx> {
            rustc_query_description! { $name<$tcx> }

            type Cache = query_storage::$name<$tcx>;

            #[inline(always)]
            fn query_state<'a>(tcx: QueryCtxt<$tcx>) -> &'a QueryState<Self::Key>
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

            fn handle_cycle_error(tcx: QueryCtxt<'tcx>, mut error: rustc_errors::DiagnosticBuilder<'_>) -> Self::Value {
                handle_cycle_error!([$($modifiers)*][tcx, error])
            }

            fn compute(tcx: QueryCtxt<'tcx>, _key: &Self::Key) -> fn(TyCtxt<'tcx>, Self::Key) -> Self::Value {
                get_provider!([$($modifiers)*][tcx, $name, _key])
            }
        })*
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
            extern_providers: Box<ExternProviders>,

            jobs: AtomicU64,

            $($(#[$attr])*  $name: QueryState<query_keys::$name<$tcx>>,)*
        }

        impl<$tcx> Queries<$tcx> {
            pub fn new(
                local_providers: Providers,
                extern_providers: ExternProviders,
            ) -> Self {
                Queries {
                    local_providers: Box::new(local_providers),
                    extern_providers: Box::new(extern_providers),
                    jobs: AtomicU64::new(1),
                    $($name: Default::default()),*
                }
            }

            pub(crate) fn try_collect_active_jobs(
                &$tcx self,
                tcx: TyCtxt<$tcx>,
            ) -> Option<QueryMap> {
                let tcx = QueryCtxt { tcx, queries: self };
                let mut jobs = QueryMap::default();

                $(
                    self.$name.try_collect_active_jobs(
                        tcx,
                        make_query::$name,
                        &mut jobs,
                    )?;
                )*

                Some(jobs)
            }
        }

        impl<'tcx> QueryEngine<'tcx> for Queries<'tcx> {
            fn as_any(&'tcx self) -> &'tcx dyn std::any::Any {
                let this = unsafe { std::mem::transmute::<&Queries<'_>, &Queries<'_>>(self) };
                this as _
            }

            $($(#[$attr])*
            #[inline(always)]
            fn $name(
                &'tcx self,
                tcx: TyCtxt<$tcx>,
                span: Span,
                key: query_keys::$name<$tcx>,
                lookup: QueryLookup,
            ) -> query_stored::$name<$tcx> {
                opt_remap_env_constness!([$($modifiers)*][key]);
                let qcx = QueryCtxt { tcx, queries: self };
                get_query::<queries::$name<$tcx>, _>(qcx, span, key, lookup)
            })*
        }
    };
}
