//! Query configuration and description traits.

use crate::dep_graph::{DepNode, DepNodeParams, SerializedDepNodeIndex};
use crate::error::HandleCycleError;
use crate::ich::StableHashingContext;
use crate::query::caches::QueryCache;
use crate::query::{QueryContext, QueryState};

use rustc_data_structures::fingerprint::Fingerprint;
use std::fmt::Debug;
use std::hash::Hash;

pub type HashResult<Qcx, Q> =
    Option<fn(&mut StableHashingContext<'_>, &<Q as QueryConfig<Qcx>>::Value) -> Fingerprint>;

pub type TryLoadFromDisk<Qcx, Q> =
    Option<fn(Qcx, SerializedDepNodeIndex) -> Option<<Q as QueryConfig<Qcx>>::Value>>;

pub trait QueryConfig<Qcx: QueryContext> {
    const NAME: &'static str;

    type Key: DepNodeParams<Qcx::DepContext> + Eq + Hash + Clone + Debug;
    type Value: Debug;
    type Stored: Debug + Copy + std::borrow::Borrow<Self::Value>;

    type Cache: QueryCache<Key = Self::Key, Stored = Self::Stored, Value = Self::Value>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(tcx: Qcx) -> &'a QueryState<Self::Key, Qcx::DepKind>
    where
        Qcx: 'a;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: Qcx) -> &'a Self::Cache
    where
        Qcx: 'a;

    fn cache_on_disk(tcx: Qcx::DepContext, key: &Self::Key) -> bool;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn execute_query(tcx: Qcx::DepContext, k: Self::Key) -> Self::Stored;

    fn compute(tcx: Qcx, key: &Self::Key) -> fn(Qcx::DepContext, Self::Key) -> Self::Value;

    fn try_load_from_disk(qcx: Qcx, idx: &Self::Key) -> TryLoadFromDisk<Qcx, Self>;

    const ANON: bool;
    const EVAL_ALWAYS: bool;
    const DEPTH_LIMIT: bool;
    const FEEDABLE: bool;

    const DEP_KIND: Qcx::DepKind;
    const HANDLE_CYCLE_ERROR: HandleCycleError;

    const HASH_RESULT: HashResult<Qcx, Self>;

    // Just here for convernience and checking that the key matches the kind, don't override this.
    fn construct_dep_node(tcx: Qcx::DepContext, key: &Self::Key) -> DepNode<Qcx::DepKind> {
        DepNode::construct(tcx, Self::DEP_KIND, key)
    }
}
