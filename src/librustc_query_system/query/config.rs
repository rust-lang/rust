//! Query configuration and description traits.

use crate::dep_graph::DepNode;
use crate::dep_graph::SerializedDepNodeIndex;
use crate::query::caches::QueryCache;
use crate::query::plumbing::CycleError;
use crate::query::{QueryContext, QueryState};
use rustc_data_structures::profiling::ProfileCategory;
use rustc_span::def_id::DefId;

use rustc_data_structures::fingerprint::Fingerprint;
use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;

// The parameter `CTX` is required in librustc_middle:
// implementations may need to access the `'tcx` lifetime in `CTX = TyCtxt<'tcx>`.
pub trait QueryConfig<CTX> {
    const NAME: &'static str;
    const CATEGORY: ProfileCategory;

    type Key: Eq + Hash + Clone + Debug;
    type Value;
    type Stored: Clone;
}

pub trait QueryAccessors<CTX: QueryContext>: QueryConfig<CTX> {
    const ANON: bool;
    const EVAL_ALWAYS: bool;
    const DEP_KIND: CTX::DepKind;

    type Cache: QueryCache<Key = Self::Key, Stored = Self::Stored, Value = Self::Value>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(tcx: CTX) -> &'a QueryState<CTX, Self::Cache>;

    fn to_dep_node(tcx: CTX, key: &Self::Key) -> DepNode<CTX::DepKind>;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn compute(tcx: CTX, key: Self::Key) -> Self::Value;

    fn hash_result(
        hcx: &mut CTX::StableHashingContext,
        result: &Self::Value,
    ) -> Option<Fingerprint>;

    fn handle_cycle_error(tcx: CTX, error: CycleError<CTX::Query>) -> Self::Value;
}

pub trait QueryDescription<CTX: QueryContext>: QueryAccessors<CTX> {
    fn describe(tcx: CTX, key: Self::Key) -> Cow<'static, str>;

    #[inline]
    fn cache_on_disk(_: CTX, _: Self::Key, _: Option<&Self::Value>) -> bool {
        false
    }

    fn try_load_from_disk(_: CTX, _: SerializedDepNodeIndex) -> Option<Self::Value> {
        panic!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}

impl<CTX: QueryContext, M> QueryDescription<CTX> for M
where
    M: QueryAccessors<CTX, Key = DefId>,
{
    default fn describe(tcx: CTX, def_id: DefId) -> Cow<'static, str> {
        if !tcx.verbose() {
            format!("processing `{}`", tcx.def_path_str(def_id)).into()
        } else {
            let name = ::std::any::type_name::<M>();
            format!("processing {:?} with query `{}`", def_id, name).into()
        }
    }

    default fn cache_on_disk(_: CTX, _: Self::Key, _: Option<&Self::Value>) -> bool {
        false
    }

    default fn try_load_from_disk(_: CTX, _: SerializedDepNodeIndex) -> Option<Self::Value> {
        panic!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}
