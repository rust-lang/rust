use std::ops::Deref;

use rustc_data_structures::sync::{AtomicU64, WorkerLocal};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::hir_id::OwnerId;
use rustc_macros::HashStable;
use rustc_query_system::HandleCycleError;
use rustc_query_system::dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
pub(crate) use rustc_query_system::query::QueryJobId;
use rustc_query_system::query::*;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::dep_graph;
use crate::dep_graph::DepKind;
use crate::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex, OnDiskCache};
use crate::query::{
    DynamicQueries, ExternProviders, Providers, QueryArenas, QueryCaches, QueryEngine, QueryStates,
};
use crate::ty::TyCtxt;

pub struct DynamicQuery<'tcx, C: QueryCache> {
    pub name: &'static str,
    pub eval_always: bool,
    pub dep_kind: DepKind,
    pub handle_cycle_error: HandleCycleError,
    // Offset of this query's state field in the QueryStates struct
    pub query_state: usize,
    // Offset of this query's cache field in the QueryCaches struct
    pub query_cache: usize,
    pub cache_on_disk: fn(tcx: TyCtxt<'tcx>, key: &C::Key) -> bool,
    pub execute_query: fn(tcx: TyCtxt<'tcx>, k: C::Key) -> C::Value,
    pub compute: fn(tcx: TyCtxt<'tcx>, key: C::Key) -> C::Value,
    pub can_load_from_disk: bool,
    pub try_load_from_disk: fn(
        tcx: TyCtxt<'tcx>,
        key: &C::Key,
        prev_index: SerializedDepNodeIndex,
        index: DepNodeIndex,
    ) -> Option<C::Value>,
    pub loadable_from_disk:
        fn(tcx: TyCtxt<'tcx>, key: &C::Key, index: SerializedDepNodeIndex) -> bool,
    pub hash_result: HashResult<C::Value>,
    pub value_from_cycle_error:
        fn(tcx: TyCtxt<'tcx>, cycle_error: &CycleError, guar: ErrorGuaranteed) -> C::Value,
    pub format_value: fn(&C::Value) -> String,
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
    pub dynamic_queries: DynamicQueries<'tcx>,

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

#[inline(always)]
pub fn query_get_at<'tcx, Cache>(
    tcx: TyCtxt<'tcx>,
    execute_query: fn(TyCtxt<'tcx>, Span, Cache::Key, QueryMode) -> Option<Cache::Value>,
    query_cache: &Cache,
    span: Span,
    key: Cache::Key,
) -> Cache::Value
where
    Cache: QueryCache,
{
    let key = key.into_query_param();
    match try_get_cached(tcx, query_cache, &key) {
        Some(value) => value,
        None => execute_query(tcx, span, key, QueryMode::Get).unwrap(),
    }
}

#[inline]
pub fn query_ensure<'tcx, Cache>(
    tcx: TyCtxt<'tcx>,
    execute_query: fn(TyCtxt<'tcx>, Span, Cache::Key, QueryMode) -> Option<Cache::Value>,
    query_cache: &Cache,
    key: Cache::Key,
    check_cache: bool,
) where
    Cache: QueryCache,
{
    let key = key.into_query_param();
    if try_get_cached(tcx, query_cache, &key).is_none() {
        execute_query(tcx, DUMMY_SP, key, QueryMode::Ensure { check_cache });
    }
}

#[inline]
pub fn query_ensure_error_guaranteed<'tcx, Cache, T>(
    tcx: TyCtxt<'tcx>,
    execute_query: fn(TyCtxt<'tcx>, Span, Cache::Key, QueryMode) -> Option<Cache::Value>,
    query_cache: &Cache,
    key: Cache::Key,
    check_cache: bool,
) -> Result<(), ErrorGuaranteed>
where
    Cache: QueryCache<Value = super::erase::Erase<Result<T, ErrorGuaranteed>>>,
    Result<T, ErrorGuaranteed>: EraseType,
{
    let key = key.into_query_param();
    if let Some(res) = try_get_cached(tcx, query_cache, &key) {
        super::erase::restore(res).map(drop)
    } else {
        execute_query(tcx, DUMMY_SP, key, QueryMode::Ensure { check_cache })
            .map(super::erase::restore)
            .map(|res| res.map(drop))
            // Either we actually executed the query, which means we got a full `Result`,
            // or we can just assume the query succeeded, because it was green in the
            // incremental cache. If it is green, that means that the previous compilation
            // that wrote to the incremental cache compiles successfully. That is only
            // possible if the cache entry was `Ok(())`, so we emit that here, without
            // actually encoding the `Result` in the cache or loading it from there.
            .unwrap_or(Ok(()))
    }
}

macro_rules! query_ensure {
    ([]$($args:tt)*) => {
        query_ensure($($args)*)
    };
    ([(return_result_from_ensure_ok) $($rest:tt)*]$($args:tt)*) => {
        query_ensure_error_guaranteed($($args)*).map(|_| ())
    };
    ([$other:tt $($modifiers:tt)*]$($args:tt)*) => {
        query_ensure!([$($modifiers)*]$($args)*)
    };
}

macro_rules! query_helper_param_ty {
    (DefId) => { impl IntoQueryParam<DefId> };
    (LocalDefId) => { impl IntoQueryParam<LocalDefId> };
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
        <$($K)* as AsLocalKey>::LocalKey
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
            queries::$name::Key<'tcx>,
        ) -> queries::$name::ProvidedValue<'tcx>
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
     $($(#[$attr:meta])*
        [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {

        #[allow(unused_lifetimes)]
        pub mod queries {
            $(pub mod $name {
                use super::super::*;

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

                /// This function takes `ProvidedValue` and coverts it to an erased `Value` by
                /// allocating it on an arena if the query has the `arena_cache` modifier. The
                /// value is then erased and returned. This will happen when computing the query
                /// using a provider or decoding a stored result.
                #[inline(always)]
                pub fn provided_to_erased<'tcx>(
                    _tcx: TyCtxt<'tcx>,
                    value: ProvidedValue<'tcx>,
                ) -> Erase<Value<'tcx>> {
                    erase(query_if_arena!([$($modifiers)*]
                        {
                            use $crate::query::arena_cached::ArenaCached;

                            if mem::needs_drop::<<$V as ArenaCached<'tcx>>::Allocated>() {
                                <$V as ArenaCached>::alloc_in_arena(
                                    |v| _tcx.query_system.arenas.$name.alloc(v),
                                    value,
                                )
                            } else {
                                <$V as ArenaCached>::alloc_in_arena(
                                    |v| _tcx.arena.dropless.alloc(v),
                                    value,
                                )
                            }
                        }
                        (value)
                    ))
                }

                pub type Storage<'tcx> = <$($K)* as keys::Key>::Cache<Erase<$V>>;

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
        }

        pub struct QueryArenas<'tcx> {
            $($(#[$attr])* pub $name: query_if_arena!([$($modifiers)*]
                (TypedArena<<$V as $crate::query::arena_cached::ArenaCached<'tcx>>::Allocated>)
                ()
            ),)*
        }

        impl Default for QueryArenas<'_> {
            fn default() -> Self {
                Self {
                    $($name: query_if_arena!([$($modifiers)*]
                        (Default::default())
                        ()
                    ),)*
                }
            }
        }

        #[derive(Default)]
        pub struct QueryCaches<'tcx> {
            $($(#[$attr])* pub $name: queries::$name::Storage<'tcx>,)*
        }

        impl<'tcx> TyCtxtEnsureOk<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(
                self,
                key: query_helper_param_ty!($($K)*),
            ) -> ensure_ok_result!([$($modifiers)*]) {
                query_ensure!(
                    [$($modifiers)*]
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    key.into_query_param(),
                    false,
                )
            })*
        }

        impl<'tcx> TyCtxtEnsureDone<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) {
                query_ensure(
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    key.into_query_param(),
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

        impl<'tcx> TyCtxtAt<'tcx> {
            $($(#[$attr])*
            #[inline(always)]
            pub fn $name(self, key: query_helper_param_ty!($($K)*)) -> $V
            {
                restore::<$V>(query_get_at(
                    self.tcx,
                    self.tcx.query_system.fns.engine.$name,
                    &self.tcx.query_system.caches.$name,
                    self.span,
                    key.into_query_param(),
                ))
            })*
        }

        pub struct DynamicQueries<'tcx> {
            $(
                pub $name: DynamicQuery<'tcx, queries::$name::Storage<'tcx>>,
            )*
        }

        #[derive(Default)]
        pub struct QueryStates<'tcx> {
            $(
                pub $name: QueryState<$($K)*, QueryStackDeferred<'tcx>>,
            )*
        }

        pub struct Providers {
            $(pub $name: for<'tcx> fn(
                TyCtxt<'tcx>,
                queries::$name::LocalKey<'tcx>,
            ) -> queries::$name::ProvidedValue<'tcx>,)*
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
                queries::$name::Key<'tcx>,
                QueryMode,
            ) -> Option<Erase<$V>>,)*
        }
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

macro_rules! define_feedable {
    ($($(#[$attr:meta])* [$($modifiers:tt)*] fn $name:ident($($K:tt)*) -> $V:ty,)*) => {
        $(impl<'tcx, K: IntoQueryParam<$($K)*> + Copy> TyCtxtFeed<'tcx, K> {
            $(#[$attr])*
            #[inline(always)]
            pub fn $name(self, value: queries::$name::ProvidedValue<'tcx>) {
                let key = self.key().into_query_param();

                let tcx = self.tcx;
                let erased = queries::$name::provided_to_erased(tcx, value);
                let value = restore::<$V>(erased);
                let cache = &tcx.query_system.caches.$name;

                let hasher: Option<fn(&mut StableHashingContext<'_>, &_) -> _> = hash_result!([$($modifiers)*]);
                match try_get_cached(tcx, cache, &key) {
                    Some(old) => {
                        let old = restore::<$V>(old);
                        if let Some(hasher) = hasher {
                            let (value_hash, old_hash): (Fingerprint, Fingerprint) = tcx.with_stable_hashing_context(|mut hcx|
                                (hasher(&mut hcx, &value), hasher(&mut hcx, &old))
                            );
                            if old_hash != value_hash {
                                // We have an inconsistency. This can happen if one of the two
                                // results is tainted by errors. In this case, delay a bug to
                                // ensure compilation is doomed, and keep the `old` value.
                                tcx.dcx().delayed_bug(format!(
                                    "Trying to feed an already recorded value for query {} key={key:?}:\n\
                                    old value: {old:?}\nnew value: {value:?}",
                                    stringify!($name),
                                ));
                            }
                        } else {
                            // The query is `no_hash`, so we have no way to perform a sanity check.
                            // If feeding the same value multiple times needs to be supported,
                            // the query should not be marked `no_hash`.
                            bug!(
                                "Trying to feed an already recorded value for query {} key={key:?}:\nold value: {old:?}\nnew value: {value:?}",
                                stringify!($name),
                            )
                        }
                    }
                    None => {
                        let dep_node = dep_graph::DepNode::construct(tcx, dep_graph::dep_kinds::$name, &key);
                        let dep_node_index = tcx.dep_graph.with_feed_task(
                            dep_node,
                            tcx,
                            &value,
                            hash_result!([$($modifiers)*]),
                        );
                        cache.complete(key, erased, dep_node_index);
                    }
                }
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
// Queries marked with `fatal_cycle` do not need the latter implementation,
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

pub use sealed::IntoQueryParam;

use super::erase::EraseType;

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
