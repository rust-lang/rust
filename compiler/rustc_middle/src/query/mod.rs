use rustc_hir::def_id::LocalDefId;

pub use self::caches::{DefIdCache, DefaultCache, QueryCache, SingleCache, VecCache};
pub use self::into_query_key::IntoQueryKey;
pub use self::job::{QueryJob, QueryJobId, QueryLatch, QueryWaiter};
pub use self::keys::{AsLocalQueryKey, LocalCrate, QueryKey};
pub use self::plumbing::{
    ActiveKeyStatus, Cycle, EnsureMode, QueryHashHelper, QueryHelper, QueryMode, QueryState,
    QuerySystem, QueryVTable, TyCtxtAt, TyCtxtEnsureDone, TyCtxtEnsureOk, TyCtxtEnsureResult,
};
pub use self::stack::QueryStackFrame;
pub use crate::queries::Providers;
use crate::ty::TyCtxt;

pub(crate) mod arena_cached;
mod caches;
pub mod erase;
pub mod impl_;
pub(crate) mod inner;
mod into_query_key;
mod job;
mod keys;
pub(crate) mod modifiers;
pub mod on_disk_cache;
pub(crate) mod plumbing;
mod stack;

pub fn describe_as_module(def_id: impl Into<LocalDefId>, tcx: TyCtxt<'_>) -> String {
    let def_id = def_id.into();
    if def_id.is_top_level_module() {
        "top-level module".to_string()
    } else {
        format!("module `{}`", tcx.def_path_str(def_id))
    }
}
