use std::ops::Deref;

use rustc_data_structures::sync::{AtomicU64, WorkerLocal};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::hir_id::OwnerId;
use rustc_macros::HashStable;
use rustc_query_system::dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
pub(crate) use rustc_query_system::query::QueryJobId;
use rustc_query_system::query::{CycleError, CycleErrorHandling, HashResult, QueryCache};
use rustc_span::{ErrorGuaranteed, Span};
pub use sealed::IntoQueryParam;

use crate::dep_graph;
use crate::dep_graph::DepKind;
use crate::query::on_disk_cache::{CacheEncoder, EncodedDepNodeIndex, OnDiskCache};
use crate::query::{
    ExternProviders, PerQueryVTables, Providers, QueryArenas, QueryCaches, QueryEngine, QueryStates,
};
use crate::ty::TyCtxt;

pub type WillCacheOnDiskForKeyFn<'tcx, Key> = fn(tcx: TyCtxt<'tcx>, key: &Key) -> bool;

pub type TryLoadFromDiskFn<'tcx, Key, Value> = fn(
    tcx: TyCtxt<'tcx>,
    key: &Key,
    prev_index: SerializedDepNodeIndex,
    index: DepNodeIndex,
) -> Option<Value>;

pub type IsLoadableFromDiskFn<'tcx, Key> =
    fn(tcx: TyCtxt<'tcx>, key: &Key, index: SerializedDepNodeIndex) -> bool;

/// Stores function pointers and other metadata for a particular query.
///
/// Used indirectly by query plumbing in `rustc_query_system`, via a trait.
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
    pub execute_query: fn(tcx: TyCtxt<'tcx>, k: C::Key) -> C::Value,
    pub compute: fn(tcx: TyCtxt<'tcx>, key: C::Key) -> C::Value,
    pub try_load_from_disk_fn: Option<TryLoadFromDiskFn<'tcx, C::Key, C::Value>>,
    pub is_loadable_from_disk_fn: Option<IsLoadableFromDiskFn<'tcx, C::Key>>,
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
