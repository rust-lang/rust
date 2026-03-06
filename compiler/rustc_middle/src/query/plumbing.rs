use std::fmt;
use std::ops::Deref;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::hash_table::HashTable;
use rustc_data_structures::sharded::Sharded;
use rustc_data_structures::sync::{AtomicU64, WorkerLocal};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::hir_id::OwnerId;
use rustc_span::{ErrorGuaranteed, Span};
pub use sealed::IntoQueryParam;

use crate::dep_graph::{DepKind, DepNodeIndex, SerializedDepNodeIndex};
use crate::ich::StableHashingContext;
use crate::queries::{ExternProviders, Providers, QueryArenas, QueryVTables, TaggedQueryKey};
use crate::query::on_disk_cache::OnDiskCache;
use crate::query::stack::QueryStackFrame;
use crate::query::{QueryCache, QueryInfo, QueryJob};
use crate::ty::TyCtxt;

/// For a particular query, keeps track of "active" keys, i.e. keys whose
/// evaluation has started but has not yet finished successfully.
///
/// (Successful query evaluation for a key is represented by an entry in the
/// query's in-memory cache.)
pub struct QueryState<'tcx, K> {
    pub active: Sharded<HashTable<(K, ActiveKeyStatus<'tcx>)>>,
}

impl<'tcx, K> Default for QueryState<'tcx, K> {
    fn default() -> QueryState<'tcx, K> {
        QueryState { active: Default::default() }
    }
}

/// For a particular query and key, tracks the status of a query evaluation
/// that has started, but has not yet finished successfully.
///
/// (Successful query evaluation for a key is represented by an entry in the
/// query's in-memory cache.)
pub enum ActiveKeyStatus<'tcx> {
    /// Some thread is already evaluating the query for this key.
    ///
    /// The enclosed [`QueryJob`] can be used to wait for it to finish.
    Started(QueryJob<'tcx>),

    /// The query panicked. Queries trying to wait on this will raise a fatal error which will
    /// silently panic.
    Poisoned,
}

/// How a particular query deals with query cycle errors.
///
/// Inspected by the code that actually handles cycle errors, to decide what
/// approach to use.
#[derive(Copy, Clone)]
pub enum CycleErrorHandling {
    Error,
    DelayBug,
}

#[derive(Clone, Debug)]
pub struct CycleError<'tcx> {
    /// The query and related span that uses the cycle.
    pub usage: Option<(Span, QueryStackFrame<'tcx>)>,
    pub cycle: Vec<QueryInfo<'tcx>>,
}

#[derive(Debug)]
pub enum QueryMode {
    /// This is a normal query call to `tcx.$query(..)` or `tcx.at(span).$query(..)`.
    Get,
    /// This is a call to `tcx.ensure_ok().$query(..)` or `tcx.ensure_done().$query(..)`.
    Ensure { ensure_mode: EnsureMode },
}

/// Distinguishes between `tcx.ensure_ok()` and `tcx.ensure_done()` in shared
/// code paths that handle both modes.
#[derive(Debug)]
pub enum EnsureMode {
    /// Corresponds to [`TyCtxt::ensure_ok`].
    Ok,
    /// Corresponds to [`TyCtxt::ensure_done`].
    Done,
}

/// Stores data and metadata (e.g. function pointers) for a particular query.
pub struct QueryVTable<'tcx, C: QueryCache> {
    pub name: &'static str,

    /// True if this query has the `anon` modifier.
    pub anon: bool,
    /// True if this query has the `eval_always` modifier.
    pub eval_always: bool,
    /// True if this query has the `depth_limit` modifier.
    pub depth_limit: bool,
    /// True if this query has the `feedable` modifier.
    pub feedable: bool,

    pub dep_kind: DepKind,
    /// How this query deals with query cycle errors.
    pub cycle_error_handling: CycleErrorHandling,
    pub state: QueryState<'tcx, C::Key>,
    pub cache: C,

    /// Function pointer that actually calls this query's provider.
    /// Also performs some associated secondary tasks; see the macro-defined
    /// implementation in `mod invoke_provider_fn` for more details.
    ///
    /// This should be the only code that calls the provider function.
    pub invoke_provider_fn: fn(tcx: TyCtxt<'tcx>, key: C::Key) -> C::Value,

    pub will_cache_on_disk_for_key_fn: fn(tcx: TyCtxt<'tcx>, key: C::Key) -> bool,

    pub try_load_from_disk_fn: fn(
        tcx: TyCtxt<'tcx>,
        key: C::Key,
        prev_index: SerializedDepNodeIndex,
        index: DepNodeIndex,
    ) -> Option<C::Value>,

    pub is_loadable_from_disk_fn:
        fn(tcx: TyCtxt<'tcx>, key: C::Key, index: SerializedDepNodeIndex) -> bool,

    /// Function pointer that hashes this query's result values.
    ///
    /// For `no_hash` queries, this function pointer is None.
    pub hash_value_fn: Option<fn(&mut StableHashingContext<'_>, &C::Value) -> Fingerprint>,

    pub value_from_cycle_error: fn(
        tcx: TyCtxt<'tcx>,
        key: C::Key,
        cycle_error: CycleError<'tcx>,
        guar: ErrorGuaranteed,
    ) -> C::Value,
    pub format_value: fn(&C::Value) -> String,

    pub create_tagged_key: fn(C::Key) -> TaggedQueryKey<'tcx>,

    /// Function pointer that is called by the query methods on [`TyCtxt`] and
    /// friends[^1], after they have checked the in-memory cache and found no
    /// existing value for this key.
    ///
    /// Transitive responsibilities include trying to load a disk-cached value
    /// if possible (incremental only), invoking the query provider if necessary,
    /// and putting the obtained value into the in-memory cache.
    ///
    /// [^1]: [`TyCtxt`], [`TyCtxtAt`], [`TyCtxtEnsureOk`], [`TyCtxtEnsureDone`]
    pub execute_query_fn: fn(TyCtxt<'tcx>, Span, C::Key, QueryMode) -> Option<C::Value>,
}

impl<'tcx, C: QueryCache> fmt::Debug for QueryVTable<'tcx, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // When debug-printing a query vtable (e.g. for ICE or tracing),
        // just print the query name to know what query we're dealing with.
        // The other fields and flags are probably just unhelpful noise.
        //
        // If there is need for a more detailed dump of all flags and fields,
        // consider writing a separate dump method and calling it explicitly.
        f.write_str(self.name)
    }
}

pub struct QuerySystem<'tcx> {
    pub arenas: WorkerLocal<QueryArenas<'tcx>>,
    pub query_vtables: QueryVTables<'tcx>,

    /// This provides access to the incremental compilation on-disk cache for query results.
    /// Do not access this directly. It is only meant to be used by
    /// `DepGraph::try_mark_green()` and the query infrastructure.
    /// This is `None` if we are not incremental compilation mode
    pub on_disk_cache: Option<OnDiskCache>,

    pub local_providers: Providers,
    pub extern_providers: ExternProviders,

    pub jobs: AtomicU64,
}

#[derive(Copy, Clone)]
pub struct TyCtxtAt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub span: Span,
}

impl<'tcx> Deref for TyCtxtAt<'tcx> {
    type Target = TyCtxt<'tcx>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.tcx
    }
}

#[derive(Copy, Clone)]
#[must_use]
pub struct TyCtxtEnsureOk<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

#[derive(Copy, Clone)]
#[must_use]
pub struct TyCtxtEnsureResult<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

#[derive(Copy, Clone)]
#[must_use]
pub struct TyCtxtEnsureDone<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

impl<'tcx> TyCtxt<'tcx> {
    /// FIXME: `ensure_ok`'s effects are subtle. Is this comment fully accurate?
    ///
    /// Wrapper that calls queries in a special "ensure OK" mode, for callers
    /// that don't need the return value and just want to invoke a query for
    /// its potential side-effect of emitting fatal errors.
    ///
    /// This can be more efficient than a normal query call, because if the
    /// query's inputs are all green, the call can return immediately without
    /// needing to obtain a value (by decoding one from disk or by executing
    /// the query).
    ///
    /// (As with all query calls, execution is also skipped if the query result
    /// is already cached in memory.)
    ///
    /// ## WARNING
    /// A subsequent normal call to the same query might still cause it to be
    /// executed! This can occur when the inputs are all green, but the query's
    /// result is not cached on disk, so the query must be executed to obtain a
    /// return value.
    ///
    /// Therefore, this call mode is not appropriate for callers that want to
    /// ensure that the query is _never_ executed in the future.
    #[inline(always)]
    pub fn ensure_ok(self) -> TyCtxtEnsureOk<'tcx> {
        TyCtxtEnsureOk { tcx: self }
    }

    /// This is a variant of `ensure_ok` only usable with queries that return
    /// `Result<_, ErrorGuaranteed>`. Queries calls through this function will
    /// return `Result<(), ErrorGuaranteed>`. I.e. the error status is returned
    /// but nothing else. As with `ensure_ok`, this can be more efficient than
    /// a normal query call.
    #[inline(always)]
    pub fn ensure_result(self) -> TyCtxtEnsureResult<'tcx> {
        TyCtxtEnsureResult { tcx: self }
    }

    /// Wrapper that calls queries in a special "ensure done" mode, for callers
    /// that don't need the return value and just want to guarantee that the
    /// query won't be executed in the future, by executing it now if necessary.
    ///
    /// This is useful for queries that read from a [`Steal`] value, to ensure
    /// that they are executed before the query that will steal the value.
    ///
    /// Unlike [`Self::ensure_ok`], a query with all-green inputs will only be
    /// skipped if its return value is stored in the disk-cache. This is still
    /// more efficient than a regular query, because in that situation the
    /// return value doesn't necessarily need to be decoded.
    ///
    /// (As with all query calls, execution is also skipped if the query result
    /// is already cached in memory.)
    ///
    /// [`Steal`]: rustc_data_structures::steal::Steal
    #[inline(always)]
    pub fn ensure_done(self) -> TyCtxtEnsureDone<'tcx> {
        TyCtxtEnsureDone { tcx: self }
    }

    /// Returns a transparent wrapper for `TyCtxt` which uses
    /// `span` as the location of queries performed through it.
    #[inline(always)]
    pub fn at(self, span: Span) -> TyCtxtAt<'tcx> {
        TyCtxtAt { tcx: self, span }
    }
}

macro_rules! query_helper_param_ty {
    (DefId) => { impl $crate::query::IntoQueryParam<DefId> };
    (LocalDefId) => { impl $crate::query::IntoQueryParam<LocalDefId> };
    ($K:ty) => { $K };
}

macro_rules! define_callbacks {
    (
        // You might expect the key to be `$K:ty`, but it needs to be `$($K:tt)*` so that
        // `query_helper_param_ty!` can match on specific type names.
        queries {
            $(
                $(#[$attr:meta])*
                fn $name:ident($($K:tt)*) -> $V:ty
                {
                    // Search for (QMODLIST) to find all occurrences of this query modifier list.
                    anon: $anon:literal,
                    arena_cache: $arena_cache:literal,
                    cache_on_disk: $cache_on_disk:literal,
                    cycle_error_handling: $cycle_error_handling:ident,
                    depth_limit: $depth_limit:literal,
                    eval_always: $eval_always:literal,
                    feedable: $feedable:literal,
                    no_hash: $no_hash:literal,
                    returns_error_guaranteed: $returns_error_guaranteed:literal,
                    separate_provide_extern: $separate_provide_extern:literal,
                }
            )*
        }
        // Non-queries are unused here.
        non_queries { $($_:tt)* }
    ) => {
        $(
            #[allow(unused_lifetimes)]
            pub mod $name {
                use super::*;
                use $crate::query::erase::{self, Erased};

                pub type Key<'tcx> = $($K)*;
                pub type Value<'tcx> = $V;

                /// Key type used by provider functions in `local_providers`.
                /// This query has the `separate_provide_extern` modifier.
                #[cfg($separate_provide_extern)]
                pub type LocalKey<'tcx> =
                    <Key<'tcx> as $crate::query::AsLocalQueryKey>::LocalQueryKey;
                /// Key type used by provider functions in `local_providers`.
                #[cfg(not($separate_provide_extern))]
                pub type LocalKey<'tcx> = Key<'tcx>;

                /// Type returned from query providers and loaded from disk-cache.
                #[cfg($arena_cache)]
                pub type ProvidedValue<'tcx> =
                    <Value<'tcx> as $crate::query::arena_cached::ArenaCached<'tcx>>::Provided;
                /// Type returned from query providers and loaded from disk-cache.
                #[cfg(not($arena_cache))]
                pub type ProvidedValue<'tcx> = Value<'tcx>;

                /// This helper function takes a value returned by the query provider
                /// (or loaded from disk, or supplied by query feeding), allocates
                /// it in an arena if requested by the `arena_cache` modifier, and
                /// then returns an erased copy of it.
                #[inline(always)]
                pub fn provided_to_erased<'tcx>(
                    tcx: TyCtxt<'tcx>,
                    provided_value: ProvidedValue<'tcx>,
                ) -> Erased<Value<'tcx>> {
                    // For queries with the `arena_cache` modifier, store the
                    // provided value in an arena and get a reference to it.
                    #[cfg($arena_cache)]
                    let value: Value<'tcx> = {
                        use $crate::query::arena_cached::ArenaCached;
                        <Value<'tcx> as ArenaCached>::alloc_in_arena(
                            tcx,
                            &tcx.query_system.arenas.$name,
                            provided_value,
                        )
                    };

                    // Otherwise, the provided value is the value (and `tcx` is unused).
                    #[cfg(not($arena_cache))]
                    let value: Value<'tcx> = {
                        let _ = tcx;
                        provided_value
                    };

                    erase::erase_val(value)
                }

                pub type Cache<'tcx> =
                    <Key<'tcx> as $crate::query::QueryKey>::Cache<Erased<Value<'tcx>>>;

                // Ensure that keys grow no larger than 88 bytes by accident.
                // Increase this limit if necessary, but do try to keep the size low if possible
                #[cfg(target_pointer_width = "64")]
                const _: () = {
                    if size_of::<Key<'static>>() > 88 {
                        panic!("{}", concat!(
                            "the query `",
                            stringify!($name),
                            "` has a key type `",
                            stringify!($($K)*),
                            "` that is too large"
                        ));
                    }
                };

                // Ensure that values grow no larger than 64 bytes by accident.
                // Increase this limit if necessary, but do try to keep the size low if possible
                #[cfg(target_pointer_width = "64")]
                #[cfg(not(feature = "rustc_randomized_layouts"))]
                const _: () = {
                    if size_of::<Value<'static>>() > 64 {
                        panic!("{}", concat!(
                            "the query `",
                            stringify!($name),
                            "` has a value type `",
                            stringify!($V),
                            "` that is too large"
                        ));
                    }
                };
            }
        )*

        /// Holds per-query arenas for queries with the `arena_cache` modifier.
        #[derive(Default)]
        pub struct QueryArenas<'tcx> {
            $(
                // Use the `ArenaCached` helper trait to determine the arena's value type.
                #[cfg($arena_cache)]
                pub $name: TypedArena<
                    <$V as $crate::query::arena_cached::ArenaCached<'tcx>>::Allocated,
                >,
            )*
        }

        impl<'tcx> $crate::query::TyCtxtEnsureOk<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, key: query_helper_param_ty!($($K)*)) {
                    crate::query::inner::query_ensure_ok_or_done(
                        self.tcx,
                        &self.tcx.query_system.query_vtables.$name,
                        $crate::query::IntoQueryParam::into_query_param(key),
                        $crate::query::EnsureMode::Ok,
                    )
                }
            )*
        }

        // Only defined when the `ensure_result` modifier is present.
        impl<'tcx> $crate::query::TyCtxtEnsureResult<'tcx> {
            $(
                #[cfg($returns_error_guaranteed)]
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(
                    self,
                    key: query_helper_param_ty!($($K)*),
                ) -> Result<(), rustc_errors::ErrorGuaranteed> {
                    crate::query::inner::query_ensure_result(
                        self.tcx,
                        &self.tcx.query_system.query_vtables.$name,
                        $crate::query::IntoQueryParam::into_query_param(key),
                    )
                }
            )*
        }

        impl<'tcx> $crate::query::TyCtxtEnsureDone<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, key: query_helper_param_ty!($($K)*)) {
                    crate::query::inner::query_ensure_ok_or_done(
                        self.tcx,
                        &self.tcx.query_system.query_vtables.$name,
                        $crate::query::IntoQueryParam::into_query_param(key),
                        $crate::query::EnsureMode::Done,
                    );
                }
            )*
        }

        impl<'tcx> TyCtxt<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                #[must_use]
                pub fn $name(self, key: query_helper_param_ty!($($K)*)) -> $V {
                    self.at(DUMMY_SP).$name(key)
                }
            )*
        }

        impl<'tcx> $crate::query::TyCtxtAt<'tcx> {
            $(
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, key: query_helper_param_ty!($($K)*)) -> $V {
                    use $crate::query::{erase, inner};

                    erase::restore_val::<$V>(inner::query_get_at(
                        self.tcx,
                        self.span,
                        &self.tcx.query_system.query_vtables.$name,
                        $crate::query::IntoQueryParam::into_query_param(key),
                    ))
                }
            )*
        }

        $(
            #[cfg($feedable)]
            impl<'tcx, K: $crate::query::IntoQueryParam<$name::Key<'tcx>> + Copy>
                TyCtxtFeed<'tcx, K>
            {
                $(#[$attr])*
                #[inline(always)]
                pub fn $name(self, value: $name::ProvidedValue<'tcx>) {
                    let key = self.key().into_query_param();
                    let erased_value = $name::provided_to_erased(self.tcx, value);
                    $crate::query::inner::query_feed(
                        self.tcx,
                        dep_graph::DepKind::$name,
                        &self.tcx.query_system.query_vtables.$name,
                        key,
                        erased_value,
                    );
                }
            }
        )*

        /// Identifies a query by kind and key. This is in contrast to `QueryJobId` which is just a number.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Debug)]
        pub enum TaggedQueryKey<'tcx> {
            $(
                $name($name::Key<'tcx>),
            )*
        }

        impl<'tcx> TaggedQueryKey<'tcx> {
            /// Formats a human-readable description of this query and its key, as
            /// specified by the `desc` query modifier.
            ///
            /// Used when reporting query cycle errors and similar problems.
            pub fn description(&self, tcx: TyCtxt<'tcx>) -> String {
                let (name, description) = ty::print::with_no_queries!(match self {
                    $(
                        TaggedQueryKey::$name(key) => (stringify!($name), _description_fns::$name(tcx, *key)),
                    )*
                });
                if tcx.sess.verbose_internals() {
                    format!("{description} [{name:?}]")
                } else {
                    description
                }
            }

            /// Returns the default span for this query if `span` is a dummy span.
            pub fn default_span(&self, tcx: TyCtxt<'tcx>, span: Span) -> Span {
                if !span.is_dummy() {
                    return span
                }
                if let TaggedQueryKey::def_span(..) = self {
                    // The `def_span` query is used to calculate `default_span`,
                    // so exit to avoid infinite recursion.
                    return DUMMY_SP
                }
                match self {
                    $(
                        TaggedQueryKey::$name(key) => crate::query::QueryKey::default_span(key, tcx),
                    )*
                }
            }

            pub fn def_kind(&self, tcx: TyCtxt<'tcx>) -> Option<DefKind> {
                // This is used to reduce code generation as it
                // can be reused for queries with the same key type.
                fn inner<'tcx>(key: &impl crate::query::QueryKey, tcx: TyCtxt<'tcx>) -> Option<DefKind> {
                    key.key_as_def_id().and_then(|def_id| def_id.as_local()).map(|def_id| tcx.def_kind(def_id))
                }

                if let TaggedQueryKey::def_kind(..) = self {
                    // Try to avoid infinite recursion.
                    return None
                }
                match self {
                    $(
                        TaggedQueryKey::$name(key) => inner(key, tcx),
                    )*
                }
            }
        }

        /// Holds a `QueryVTable` for each query.
        pub struct QueryVTables<'tcx> {
            $(
                pub $name: ::rustc_middle::query::plumbing::QueryVTable<'tcx, $name::Cache<'tcx>>,
            )*
        }

        pub struct Providers {
            $(
                /// This is the provider for the query. Use `Find references` on this to
                /// navigate between the provider assignment and the query definition.
                pub $name: for<'tcx> fn(
                    TyCtxt<'tcx>,
                    $name::LocalKey<'tcx>,
                ) -> $name::ProvidedValue<'tcx>,
            )*
        }

        pub struct ExternProviders {
            $(
                #[cfg($separate_provide_extern)]
                pub $name: for<'tcx> fn(
                    TyCtxt<'tcx>,
                    $name::Key<'tcx>,
                ) -> $name::ProvidedValue<'tcx>,
            )*
        }

        impl Default for Providers {
            fn default() -> Self {
                Providers {
                    $(
                        $name: |_, key| {
                            $crate::query::plumbing::default_query(stringify!($name), &key)
                        },
                    )*
                }
            }
        }

        impl Default for ExternProviders {
            fn default() -> Self {
                ExternProviders {
                    $(
                        #[cfg($separate_provide_extern)]
                        $name: |_, key| $crate::query::plumbing::default_extern_query(
                            stringify!($name),
                            &key,
                        ),
                    )*
                }
            }
        }

        impl Copy for Providers {}
        impl Clone for Providers {
            fn clone(&self) -> Self { *self }
        }

        impl Copy for ExternProviders {}
        impl Clone for ExternProviders {
            fn clone(&self) -> Self { *self }
        }
    };
}

mod sealed {
    use rustc_hir::def_id::{LocalModDefId, ModDefId};

    use super::{DefId, LocalDefId, OwnerId};

    /// An analogue of the `Into` trait that's intended only for query parameters.
    ///
    /// This exists to allow queries to accept either `DefId` or `LocalDefId` while requiring that the
    /// user call `to_def_id` to convert between them everywhere else.
    pub trait IntoQueryParam<P> {
        fn into_query_param(self) -> P;
    }

    impl<P> IntoQueryParam<P> for P {
        #[inline(always)]
        fn into_query_param(self) -> P {
            self
        }
    }

    impl IntoQueryParam<LocalDefId> for OwnerId {
        #[inline(always)]
        fn into_query_param(self) -> LocalDefId {
            self.def_id
        }
    }

    impl IntoQueryParam<DefId> for LocalDefId {
        #[inline(always)]
        fn into_query_param(self) -> DefId {
            self.to_def_id()
        }
    }

    impl IntoQueryParam<DefId> for OwnerId {
        #[inline(always)]
        fn into_query_param(self) -> DefId {
            self.to_def_id()
        }
    }

    impl IntoQueryParam<DefId> for ModDefId {
        #[inline(always)]
        fn into_query_param(self) -> DefId {
            self.to_def_id()
        }
    }

    impl IntoQueryParam<DefId> for LocalModDefId {
        #[inline(always)]
        fn into_query_param(self) -> DefId {
            self.to_def_id()
        }
    }

    impl IntoQueryParam<LocalDefId> for LocalModDefId {
        #[inline(always)]
        fn into_query_param(self) -> LocalDefId {
            self.into()
        }
    }
}

#[cold]
pub(crate) fn default_query(name: &str, key: &dyn std::fmt::Debug) -> ! {
    bug!(
        "`tcx.{name}({key:?})` is not supported for this key;\n\
        hint: Queries can be either made to the local crate, or the external crate. \
        This error means you tried to use it for one that's not supported.\n\
        If that's not the case, {name} was likely never assigned to a provider function.\n",
    )
}

#[cold]
pub(crate) fn default_extern_query(name: &str, key: &dyn std::fmt::Debug) -> ! {
    bug!(
        "`tcx.{name}({key:?})` unsupported by its crate; \
         perhaps the `{name}` query was never assigned a provider function",
    )
}
