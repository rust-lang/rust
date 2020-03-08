//! Query configuration and description traits.

use crate::dep_graph::SerializedDepNodeIndex;
use crate::dep_graph::{DepKind, DepNode};
use crate::ty::query::caches::QueryCache;
use crate::ty::query::plumbing::CycleError;
use crate::ty::query::QueryState;
use crate::ty::TyCtxt;
use rustc_data_structures::profiling::ProfileCategory;
use rustc_hir::def_id::DefId;

use crate::ich::StableHashingContext;
use rustc_data_structures::fingerprint::Fingerprint;
use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;

pub trait QueryConfig<CTX> {
    const NAME: &'static str;
    const CATEGORY: ProfileCategory;

    type Key: Eq + Hash + Clone + Debug;
    type Value: Clone;
}

pub trait QueryContext: Copy {
    type Query;
}

pub(crate) trait QueryAccessors<CTX: QueryContext>: QueryConfig<CTX> {
    const ANON: bool;
    const EVAL_ALWAYS: bool;
    const DEP_KIND: DepKind;

    type Cache: QueryCache<Key = Self::Key, Value = Self::Value>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_state<'a>(tcx: CTX) -> &'a QueryState<CTX, Self::Cache>;

    fn to_dep_node(tcx: CTX, key: &Self::Key) -> DepNode;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn compute(tcx: CTX, key: Self::Key) -> Self::Value;

    fn hash_result(hcx: &mut StableHashingContext<'_>, result: &Self::Value)
    -> Option<Fingerprint>;

    fn handle_cycle_error(tcx: CTX, error: CycleError<CTX>) -> Self::Value;
}

pub(crate) trait QueryDescription<'tcx>: QueryAccessors<TyCtxt<'tcx>> {
    fn describe(tcx: TyCtxt<'_>, key: Self::Key) -> Cow<'static, str>;

    #[inline]
    fn cache_on_disk(_: TyCtxt<'tcx>, _: Self::Key, _: Option<&Self::Value>) -> bool {
        false
    }

    fn try_load_from_disk(_: TyCtxt<'tcx>, _: SerializedDepNodeIndex) -> Option<Self::Value> {
        bug!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}

impl<'tcx, M> QueryDescription<'tcx> for M
where
    M: QueryAccessors<TyCtxt<'tcx>, Key = DefId>,
    //M::Cache: QueryCache<DefId, M::Value>,
{
    default fn describe(tcx: TyCtxt<'_>, def_id: DefId) -> Cow<'static, str> {
        if !tcx.sess.verbose() {
            format!("processing `{}`", tcx.def_path_str(def_id)).into()
        } else {
            let name = ::std::any::type_name::<M>();
            format!("processing {:?} with query `{}`", def_id, name).into()
        }
    }

    default fn cache_on_disk(_: TyCtxt<'tcx>, _: Self::Key, _: Option<&Self::Value>) -> bool {
        false
    }

    default fn try_load_from_disk(
        _: TyCtxt<'tcx>,
        _: SerializedDepNodeIndex,
    ) -> Option<Self::Value> {
        bug!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}
