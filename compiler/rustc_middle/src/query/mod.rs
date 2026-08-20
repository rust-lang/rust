use rustc_hir::def_id::LocalDefId;

pub use self::caches::{DefIdCache, DefaultCache, QueryCache, SingleCache, VecCache};
pub use self::calls::{TyCtxtAt, TyCtxtEnsureDone, TyCtxtEnsureOk, TyCtxtEnsureResult};
pub use self::into_query_key::IntoQueryKey;
pub use self::job::{
    ActiveKeyStatus, QueryCycle, QueryJob, QueryJobId, QueryLatch, QueryStackFrame, QueryState,
    QueryWaiter,
};
pub use self::keys::{LocalCrate, QueryKey};
pub use self::system::{QueryMode, QuerySystem, QueryVTable};
pub use crate::queries::Providers;
use crate::ty::TyCtxt;

pub(crate) mod arena_cached;
mod caches;
pub(crate) mod calls;
pub mod erase;
mod into_query_key;
mod job;
mod keys;
pub(crate) mod modifiers;
pub mod on_disk_cache;
pub(crate) mod plumbing;
mod system;

pub fn describe_as_module(def_id: impl Into<LocalDefId>, tcx: TyCtxt<'_>) -> String {
    let def_id = def_id.into();
    if def_id.is_top_level_module() {
        "top-level module".to_string()
    } else {
        format!("module `{}`", tcx.def_path_str(def_id))
    }
}
