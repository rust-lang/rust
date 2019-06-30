use crate::dep_graph::SerializedDepNodeIndex;
use crate::dep_graph::{DepKind, DepNode};
use crate::hir::def_id::{CrateNum, DefId};
use crate::ty::TyCtxt;
use crate::ty::query::queries;
use crate::ty::query::{Query, QueryName};
use crate::ty::query::QueryCache;
use crate::ty::query::plumbing::CycleError;
use crate::util::profiling::ProfileCategory;

use std::borrow::Cow;
use std::hash::Hash;
use std::fmt::Debug;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::fingerprint::Fingerprint;
use crate::ich::StableHashingContext;

// Query configuration and description traits.

// FIXME(eddyb) false positive, the lifetime parameter is used for `Key`/`Value`.
#[allow(unused_lifetimes)]
pub trait QueryConfig<'tcx> {
    const NAME: QueryName;
    const CATEGORY: ProfileCategory;

    type Key: Eq + Hash + Clone + Debug;
    type Value: Clone;
}

pub(crate) trait QueryAccessors<'tcx>: QueryConfig<'tcx> {
    const ANON: bool;
    const EVAL_ALWAYS: bool;

    fn query(key: Self::Key) -> Query<'tcx>;

    // Don't use this method to access query results, instead use the methods on TyCtxt
    fn query_cache<'a>(tcx: TyCtxt<'tcx>) -> &'a Lock<QueryCache<'tcx, Self>>;

    fn to_dep_node(tcx: TyCtxt<'tcx>, key: &Self::Key) -> DepNode;

    fn dep_kind() -> DepKind;

    // Don't use this method to compute query results, instead use the methods on TyCtxt
    fn compute(tcx: TyCtxt<'tcx>, key: Self::Key) -> Self::Value;

    fn hash_result(
        hcx: &mut StableHashingContext<'_>,
        result: &Self::Value
    ) -> Option<Fingerprint>;

    fn handle_cycle_error(tcx: TyCtxt<'tcx>, error: CycleError<'tcx>) -> Self::Value;
}

pub(crate) trait QueryDescription<'tcx>: QueryAccessors<'tcx> {
    fn describe(tcx: TyCtxt<'_>, key: Self::Key) -> Cow<'static, str>;

    #[inline]
    fn cache_on_disk(_: TyCtxt<'tcx>, _: Self::Key, _: Option<&Self::Value>) -> bool {
        false
    }

    fn try_load_from_disk(_: TyCtxt<'tcx>, _: SerializedDepNodeIndex) -> Option<Self::Value> {
        bug!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}

impl<'tcx, M: QueryAccessors<'tcx, Key = DefId>> QueryDescription<'tcx> for M {
    default fn describe(tcx: TyCtxt<'_>, def_id: DefId) -> Cow<'static, str> {
        if !tcx.sess.verbose() {
            format!("processing `{}`", tcx.def_path_str(def_id)).into()
        } else {
            let name = unsafe { ::std::intrinsics::type_name::<M>() };
            format!("processing {:?} with query `{}`", def_id, name).into()
        }
    }
}

impl<'tcx> QueryDescription<'tcx> for queries::analysis<'tcx> {
    fn describe(_tcx: TyCtxt<'_>, _: CrateNum) -> Cow<'static, str> {
        "running analysis passes on this crate".into()
    }
}
