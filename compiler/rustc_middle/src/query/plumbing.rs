use std::fmt::Debug;
use std::ops::Deref;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::hash_table::HashTable;
use rustc_data_structures::sharded::Sharded;
use rustc_data_structures::sync::{AtomicU64, WorkerLocal};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::hir_id::OwnerId;
use rustc_macros::HashStable;
use rustc_span::{ErrorGuaranteed, Span};
pub use sealed::IntoQueryParam;

use crate::dep_graph;
use crate::dep_graph::{DepKind, DepNodeIndex, SerializedDepNodeIndex};
use crate::ich::StableHashingContext;
use crate::queries::{
    ExternProviders, PerQueryVTables, Providers, QueryArenas, QueryCaches, QueryEngine, QueryStates,
};
use crate::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex, OnDiskCache};
use crate::query::stack::{QueryStackDeferred, QueryStackFrame, QueryStackFrameExtra};
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
    Fatal,
    DelayBug,
    Stash,
}

pub type WillCacheOnDiskForKeyFn<'tcx, Key> = fn(tcx: TyCtxt<'tcx>, key: &Key) -> bool;

pub type TryLoadFromDiskFn<'tcx, Key, Value> = fn(
    tcx: TyCtxt<'tcx>,
    key: &Key,
    prev_index: SerializedDepNodeIndex,
    index: DepNodeIndex,
) -> Option<Value>;

pub type IsLoadableFromDiskFn<'tcx, Key> =
    fn(tcx: TyCtxt<'tcx>, key: &Key, index: SerializedDepNodeIndex) -> bool;

pub type HashResult<V> = Option<fn(&mut StableHashingContext<'_>, &V) -> Fingerprint>;

#[derive(Clone, Debug)]
pub struct CycleError<I = QueryStackFrameExtra> {
    /// The query and related span that uses the cycle.
    pub usage: Option<(Span, QueryStackFrame<I>)>,
    pub cycle: Vec<QueryInfo<I>>,
}

impl<'tcx> CycleError<QueryStackDeferred<'tcx>> {
    pub fn lift(&self) -> CycleError<QueryStackFrameExtra> {
        CycleError {
            usage: self.usage.as_ref().map(|(span, frame)| (*span, frame.lift())),
            cycle: self.cycle.iter().map(|info| info.lift()).collect(),
        }
    }
}

#[derive(Debug)]
pub enum QueryMode {
    Get,
    Ensure { check_cache: bool },
}

/// Stores function pointers and other metadata for a particular query.
pub struct QueryVTable<'tcx, C: QueryCache> {
    pub name: &'static str,
    pub eval_always: bool,
    pub dep_kind: DepKind,
    /// How this query deals with query cycle errors.
    pub cycle_error_handling: CycleErrorHandling,
    // Offset of this query's state field in the QueryStates struct
    pub query_state: usize,
    // Offset of this query's cache field in the QueryCaches struct
    pub query_cache: usize,
    pub will_cache_on_disk_for_key_fn: Option<WillCacheOnDiskForKeyFn<'tcx, C::Key>>,

    /// Function pointer that calls `tcx.$query(key)` for this query and
    /// discards the returned value.
    ///
    /// This is a weird thing to be doing, and probably not what you want.
    /// It is used for loading query results from disk-cache in some cases.
    pub call_query_method_fn: fn(tcx: TyCtxt<'tcx>, key: C::Key),

    /// Function pointer that actually calls this query's provider.
    /// Also performs some associated secondary tasks; see the macro-defined
    /// implementation in `mod invoke_provider_fn` for more details.
    ///
    /// This should be the only code that calls the provider function.
    pub invoke_provider_fn: fn(tcx: TyCtxt<'tcx>, key: C::Key) -> C::Value,

    pub try_load_from_disk_fn: Option<TryLoadFromDiskFn<'tcx, C::Key, C::Value>>,
    pub is_loadable_from_disk_fn: Option<IsLoadableFromDiskFn<'tcx, C::Key>>,
    pub hash_result: HashResult<C::Value>,
    pub value_from_cycle_error:
        fn(tcx: TyCtxt<'tcx>, cycle_error: &CycleError, guar: ErrorGuaranteed) -> C::Value,
    pub format_value: fn(&C::Value) -> String,

    /// Formats a human-readable description of this query and its key, as
    /// specified by the `desc` query modifier.
    ///
    /// Used when reporting query cycle errors and similar problems.
    pub description_fn: fn(TyCtxt<'tcx>, C::Key) -> String,
}

pub struct QuerySystemFns {
    pub engine: QueryEngine,
    pub local_providers: Providers,
    pub extern_providers: ExternProviders,
    pub encode_query_results: for<'tcx> fn(
        tcx: TyCtxt<'tcx>,
        encoder: &mut CacheEncoder<'_, 'tcx>,
        query_result_index: &mut EncodedDepNodeIndex,
    ),
    pub try_mark_green: for<'tcx> fn(tcx: TyCtxt<'tcx>, dep_node: &dep_graph::DepNode) -> bool,
}

pub struct QuerySystem<'tcx> {
    pub states: QueryStates<'tcx>,
    pub arenas: WorkerLocal<QueryArenas<'tcx>>,
    pub caches: QueryCaches<'tcx>,
    pub query_vtables: PerQueryVTables<'tcx>,

    /// This provides access to the incremental compilation on-disk cache for query results.
    /// Do not access this directly. It is only meant to be used by
    /// `DepGraph::try_mark_green()` and the query infrastructure.
    /// This is `None` if we are not incremental compilation mode
    pub on_disk_cache: Option<OnDiskCache>,

    pub fns: QuerySystemFns,

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
pub struct TyCtxtEnsureDone<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

impl<'tcx> TyCtxt<'tcx> {
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
    ///
    /// ## `return_result_from_ensure_ok`
    /// If a query has the `return_result_from_ensure_ok` modifier, calls via
    /// `ensure_ok` will instead return `Result<(), ErrorGuaranteed>`. If the
    /// query needs to be executed, and execution returns an error, that error
    /// is returned to the caller.
    #[inline(always)]
    pub fn ensure_ok(self) -> TyCtxtEnsureOk<'tcx> {
        TyCtxtEnsureOk { tcx: self }
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

    pub fn try_mark_green(self, dep_node: &dep_graph::DepNode) -> bool {
        (self.query_system.fns.try_mark_green)(self, dep_node)
    }
}

/// Calls either `query_ensure` or `query_ensure_error_guaranteed`, depending
/// on whether the list of modifiers contains `return_result_from_ensure_ok`.
macro_rules! query_ensure_select {
    ([]$($args:tt)*) => {
        crate::query::inner::query_ensure($($args)*)
    };
    ([(return_result_from_ensure_ok) $($rest:tt)*]$($args:tt)*) => {
        crate::query::inner::query_ensure_error_guaranteed($($args)*)
    };
    ([$other:tt $($modifiers:tt)*]$($args:tt)*) => {
        query_ensure_select!([$($modifiers)*]$($args)*)
    };
}

macro_rules! query_helper_param_ty {
    (DefId) => { impl $crate::query::IntoQueryParam<DefId> };
    (LocalDefId) => { impl $crate::query::IntoQueryParam<LocalDefId> };
    ($K:ty) => { $K };
}

macro_rules! query_if_arena {
    ([] $arena:tt $no_arena:tt) => {
        $no_arena
    };
    ([(arena_cache) $($rest:tt)*] $arena:tt $no_arena:tt) => {
        $arena
    };
    ([$other:tt $($modifiers:tt)*]$($args:tt)*) => {
        query_if_arena!([$($modifiers)*]$($args)*)
    };
}

/// If `separate_provide_extern`, then the key can be projected to its
/// local key via `<$K as AsLocalKey>::LocalKey`.
macro_rules! local_key_if_separate_extern {
    ([] $($K:tt)*) => {
        $($K)*
    };
    ([(separate_provide_extern) $($rest:tt)*] $($K:tt)*) => {
        <$($K)* as $crate::query::AsLocalKey>::LocalKey
    };
    ([$other:tt $($modifiers:tt)*] $($K:tt)*) => {
        local_key_if_separate_extern!([$($modifiers)*] $($K)*)
    };
}

macro_rules! separate_provide_extern_decl {
    ([][$name:ident]) => {
        ()
    };
    ([(separate_provide_extern) $($rest:tt)*][$name:ident]) => {
        for<'tcx> fn(
            TyCtxt<'tcx>,
            $name::Key<'tcx>,
        ) -> $name::ProvidedValue<'tcx>
    };
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        separate_provide_extern_decl!([$($modifiers)*][$($args)*])
    };
}

macro_rules! ensure_ok_result {
    ( [] ) => {
        ()
    };
    ( [(return_result_from_ensure_ok) $($rest:tt)*] ) => {
        Result<(), ErrorGuaranteed>
    };
    ( [$other:tt $($modifiers:tt)*] ) => {
        ensure_ok_result!( [$($modifiers)*] )
    };
}

macro_rules! separate_provide_extern_default {
    ([][$name:ident]) => {
        ()
    };
    ([(separate_provide_extern) $($rest:tt)*][$name:ident]) => {
        |_, key| $crate::query::plumbing::default_extern_query(stringify!($name), &key)
    };
    ([$other:tt $($modifiers:tt)*][$($args:tt)*]) => {
        separate_provide_extern_default!([$($modifiers)*][$($args)*])
    };
}

macro_rules! define_callbacks {
    (
        $(
            $(#[$attr:meta])*
            [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,
        )*
    ) => {
        $(#[allow(unused_lifetimes)] pub mod $name {
            use super::*;
            use $crate::query::erase::{self, Erased};

            pub type Key<'tcx> = $($K)*;
            pub type Value<'tcx> = $V;

            pub type LocalKey<'tcx> = local_key_if_separate_extern!([$($modifiers)*] $($K)*);

            /// This type alias specifies the type returned from query providers and the type
            /// used for decoding. For regular queries this is the declared returned type `V`,
            /// but `arena_cache` will use `<V as ArenaCached>::Provided` instead.
            pub type ProvidedValue<'tcx> = query_if_arena!(
                [$($modifiers)*]
                (<$V as $crate::query::arena_cached::ArenaCached<'tcx>>::Provided)
                ($V)
            );

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
                let value: Value<'tcx> = query_if_arena!([$($modifiers)*] {
                    <$V as $crate::query::arena_cached::ArenaCached>::alloc_in_arena(
                        tcx,
                        &tcx.query_system.arenas.$name,
                        provided_value,
                    )
                } {
                    // Otherwise, the provided value is the value (and `tcx` is unused).
                    let _ = tcx;
                    provided_value
                });
                erase::erase_val(value)
            }

            pub type Storage<'tcx> = <$($K)* as $crate::query::Key>::Cache<Erased<$V>>;

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
        })*

        /// Holds per-query arenas for queries with the `arena_cache` modifier.
        #[derive(Default)]
        pub struct QueryArenas<'tcx> {
            $(
                $(#[$attr])*
                pub $name: query_if_arena!([$($modifiers)*]
                    // Use the `ArenaCached` helper trait to determine the arena's value type.
                    (TypedArena<<$V as $crate::query::arena_cached::ArenaCached<'tcx>>::Allocated>)
                    // No arena for this query, so the field type is `()`.
                    ()
                ),
            )*
        }

        #[derive(Default)]
        pub struct QueryCaches<'tcx> {
            $($(#[$attr])* pub $name: $name::Storage<'tcx>,)*
        }

        impl<'tcx> $crate::query::TyCtxtEnsureOk<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(
                self,
                key: query_helper_param_ty!($($K)*),
            ) -> ensure_ok_result!([$($modifiers)*]) {
                query_ensure_select!(
                    [$($modifiers)*]
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    $crate::query::IntoQueryParam::into_query_param(key),
                    false,
                )
            })*
        }

        impl<'tcx> $crate::query::TyCtxtEnsureDone<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) {
                crate::query::inner::query_ensure(
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    $crate::query::IntoQueryParam::into_query_param(key),
                    true,
                );
            })*
        }

        impl<'tcx> TyCtxt<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            #[must_use]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) -> $V
            {
                self.at(DUMMY_SP).$name(key)
            })*
        }

        impl<'tcx> $crate::query::TyCtxtAt<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) -> $V
            {
                use $crate::query::{erase, inner};

                erase::restore_val::<$V>(inner::query_get_at(
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    self.span,
                    $crate::query::IntoQueryParam::into_query_param(key),
                ))
            })*
        }

        /// Holds a `QueryVTable` for each query.
        ///
        /// ("Per" just makes this pluralized name more visually distinct.)
        pub struct PerQueryVTables<'tcx> {
            $(
                pub $name: ::rustc_middle::query::plumbing::QueryVTable<'tcx, $name::Storage<'tcx>>,
            )*
        }

        #[derive(Default)]
        pub struct QueryStates<'tcx> {
            $(
                pub $name: $crate::query::QueryState<'tcx, $($K)*>,
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
            $(pub $name: separate_provide_extern_decl!([$($modifiers)*][$name]),)*
        }

        impl Default for Providers {
            fn default() -> Self {
                Providers {
                    $($name: |_, key| $crate::query::plumbing::default_query(stringify!($name), &key)),*
                }
            }
        }

        impl Default for ExternProviders {
            fn default() -> Self {
                ExternProviders {
                    $($name: separate_provide_extern_default!([$($modifiers)*][$name]),)*
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

        pub struct QueryEngine {
            $(pub $name: for<'tcx> fn(
                TyCtxt<'tcx>,
                Span,
                $name::Key<'tcx>,
                $crate::query::QueryMode,
            ) -> Option<$crate::query::erase::Erased<$V>>,)*
        }
    };
}

macro_rules! define_feedable {
    ($($(#[$attr:meta])* [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {
        $(impl<'tcx, K: $crate::query::IntoQueryParam<$($K)*> + Copy> TyCtxtFeed<'tcx, K> {
            $(#[$attr])*
            #[inline(always)]
            pub fn $name(self, value: $name::ProvidedValue<'tcx>) {
                let key = self.key().into_query_param();

                let tcx = self.tcx;
                let erased_value = $name::provided_to_erased(tcx, value);

                let dep_kind: dep_graph::DepKind = dep_graph::dep_kinds::$name;

                $crate::query::inner::query_feed(
                    tcx,
                    dep_kind,
                    &tcx.query_system.query_vtables.$name,
                    &tcx.query_system.caches.$name,
                    key,
                    erased_value,
                );
            }
        })*
    }
}

// Each of these queries corresponds to a function pointer field in the
// `Providers` struct for requesting a value of that type, and a method
// on `tcx: TyCtxt` (and `tcx.at(span)`) for doing that request in a way
// which memoizes and does dep-graph tracking, wrapping around the actual
// `Providers` that the driver creates (using several `rustc_*` crates).
//
// The result type of each query must implement `Clone`, and additionally
// `ty::query::values::Value`, which produces an appropriate placeholder
// (error) value if the query resulted in a query cycle.
// Queries marked with `cycle_fatal` do not need the latter implementation,
// as they will raise an fatal error on query cycles instead.

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

    impl<'a, P: Copy> IntoQueryParam<P> for &'a P {
        #[inline(always)]
        fn into_query_param(self) -> P {
            *self
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

#[derive(Copy, Clone, Debug, HashStable)]
pub struct CyclePlaceholder(pub ErrorGuaranteed);

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
