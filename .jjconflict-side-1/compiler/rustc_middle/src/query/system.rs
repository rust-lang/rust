use std::fmt;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::sync::{AtomicU64, Lock, WorkerLocal};
use rustc_errors::Diag;
use rustc_span::{Span, Symbol};

use crate::dep_graph::{
    DepKind, DepKindVTable, DepNodeIndex, QuerySideEffect, SerializedDepNodeIndex,
};
use crate::ich::StableHashState;
use crate::queries::{ExternProviders, Providers, QueryArenas, QueryVTables, TaggedQueryKey};
use crate::query::on_disk_cache::OnDiskCache;
use crate::query::{QueryCache, QueryCycle, QueryKey, QueryState};
use crate::ty::TyCtxt;

#[derive(Debug)]
pub enum QueryMode {
    /// This is a normal query call to `tcx.$query(..)` or `tcx.at(span).$query(..)`.
    Get,
    /// This is a call to `tcx.ensure_ok().$query(..)`.
    EnsureOk,
}

/// Stores data and metadata (e.g. function pointers) for a particular query.
pub struct QueryVTable<'tcx, C: QueryCache> {
    pub name: &'static str,

    /// True if this query has the `eval_always` modifier.
    pub eval_always: bool,
    /// True if this query has the `depth_limit` modifier.
    pub depth_limit: bool,
    /// True if this query has the `feedable` modifier.
    pub feedable: bool,

    pub cache_on_disk_local: bool,
    pub separate_provide_extern: bool,

    pub dep_kind: DepKind,
    pub state: QueryState<'tcx, C::Key>,
    pub cache: C,

    /// Function pointer that actually calls this query's provider.
    /// Also performs some associated secondary tasks; see the macro-defined
    /// implementation in `mod invoke_provider_fn` for more details.
    ///
    /// This should be the only code that calls the provider function.
    pub invoke_provider_fn: fn(tcx: TyCtxt<'tcx>, key: C::Key) -> C::Value,

    /// Function pointer that tries to load a query value from disk.
    ///
    /// This should only be called after a successful check of [`Self::will_cache_on_disk_for_key`].
    pub try_load_from_disk_fn:
        fn(tcx: TyCtxt<'tcx>, prev_index: SerializedDepNodeIndex) -> Option<C::Value>,

    /// Function pointer that hashes this query's result values.
    ///
    /// For `no_hash` queries, this function pointer is None.
    pub hash_value_fn: Option<fn(&mut StableHashState<'_>, &C::Value) -> Fingerprint>,

    /// Function pointer that handles a cycle error. `error` must be consumed, e.g. with `emit` (if
    /// it should be emitted) or `delay_as_bug` (if it need not be emitted because an alternative
    /// error is created and emitted). A value may be returned, or (more commonly) the function may
    /// just abort after emitting the error.
    pub handle_cycle_error_fn:
        fn(tcx: TyCtxt<'tcx>, key: C::Key, cycle: QueryCycle<'tcx>, error: Diag<'_>) -> C::Value,

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
    /// [^1]: [`TyCtxt`], [`crate::query::TyCtxtAt`], [`crate::query::TyCtxtEnsureOk`],
    /// [`crate::query::TyCtxtEnsureDone`]
    pub execute_query_fn: fn(TyCtxt<'tcx>, Span, C::Key, QueryMode) -> Option<C::Value>,
}

impl<'tcx, C: QueryCache> QueryVTable<'tcx, C> {
    pub fn will_cache_on_disk_for_key(&self, key: C::Key) -> bool {
        self.cache_on_disk_local && (!self.separate_provide_extern || key.as_local_key().is_some())
    }
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
    pub dep_kind_vtables: &'tcx [DepKindVTable<'tcx>],
    pub query_vtables: QueryVTables<'tcx>,

    /// Side-effect associated with each [`DepKind::SideEffect`] node in the
    /// current incremental-compilation session. Side effects will be written
    /// to disk, and loaded by [`OnDiskCache`] in the next session.
    ///
    /// Always empty if incremental compilation is off.
    pub side_effects: Lock<FxIndexMap<DepNodeIndex, QuerySideEffect>>,

    /// Enabled features that are used in the current compilation.
    ///
    /// The value is the `DepNodeIndex` of the node that encodes the used feature.
    pub used_features: Lock<FxHashMap<Symbol, DepNodeIndex>>,

    /// This provides access to the incremental compilation on-disk cache for query results.
    /// Do not access this directly. It is only meant to be used by
    /// `DepGraph::try_mark_green()` and the query infrastructure.
    /// This is `None` if we are not incremental compilation mode
    pub on_disk_cache: Option<OnDiskCache>,

    pub local_providers: Providers,
    pub extern_providers: ExternProviders,

    pub jobs: AtomicU64,

    pub cycle_handler_nesting: Lock<u8>,
}
