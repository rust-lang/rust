//! Query configuration and description traits.

use crate::dep_graph::SerializedDepNodeIndex;
use crate::ty::query::caches::QueryCache;
use crate::ty::query::job::{QueryJobId, QueryJobInfo};
use crate::ty::query::plumbing::CycleError;
use crate::ty::query::QueryState;
use rustc_data_structures::profiling::ProfileCategory;
use rustc_hir::def_id::DefId;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_query_system::dep_graph::{DepContext, DepNode};
use rustc_session::Session;
use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;

pub trait QueryConfig<CTX> {
    const NAME: &'static str;
    const CATEGORY: ProfileCategory;

    type Key: Eq + Hash + Clone + Debug;
    type Value: Clone;
}

pub trait QueryContext: DepContext {
    type Query: Clone;

    /// Access the session.
    fn session(&self) -> &Session;

    /// Get string representation from DefPath.
    fn def_path_str(&self, def_id: DefId) -> String;

    /// Get the query information from the TLS context.
    fn read_query_job<R>(&self, op: impl FnOnce(Option<QueryJobId<Self::DepKind>>) -> R) -> R;

    fn try_collect_active_jobs(
        &self,
    ) -> Option<FxHashMap<QueryJobId<Self::DepKind>, QueryJobInfo<Self>>>;
}

pub(crate) trait QueryAccessors<CTX: QueryContext>: QueryConfig<CTX> {
    const ANON: bool;
    const EVAL_ALWAYS: bool;
    const DEP_KIND: CTX::DepKind;

    type Cache: QueryCache<CTX, Key = Self::Key, Value = Self::Value>;

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

pub(crate) trait QueryDescription<CTX: QueryContext>: QueryAccessors<CTX> {
    fn describe(tcx: CTX, key: Self::Key) -> Cow<'static, str>;

    #[inline]
    fn cache_on_disk(_: CTX, _: Self::Key, _: Option<&Self::Value>) -> bool {
        false
    }

    fn try_load_from_disk(_: CTX, _: SerializedDepNodeIndex) -> Option<Self::Value> {
        bug!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}

impl<CTX: QueryContext, M> QueryDescription<CTX> for M
where
    M: QueryAccessors<CTX, Key = DefId>,
{
    default fn describe(tcx: CTX, def_id: DefId) -> Cow<'static, str> {
        if !tcx.session().verbose() {
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
        bug!("QueryDescription::load_from_disk() called for an unsupported query.")
    }
}
