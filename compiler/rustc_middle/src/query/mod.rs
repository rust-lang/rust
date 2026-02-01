use rustc_hir::def_id::LocalDefId;

pub use self::keys::{AsLocalKey, Key, LocalCrate};
pub use self::plumbing::{IntoQueryParam, TyCtxtAt, TyCtxtEnsureDone, TyCtxtEnsureOk};
pub use crate::queries::Providers;
use crate::ty::TyCtxt;

pub(crate) mod arena_cached;
pub mod erase;
pub(crate) mod inner;
mod keys;
pub mod on_disk_cache;
#[macro_use]
pub mod plumbing;

pub fn describe_as_module(def_id: impl Into<LocalDefId>, tcx: TyCtxt<'_>) -> String {
    let def_id = def_id.into();
    if def_id.is_top_level_module() {
        "top-level module".to_string()
    } else {
        format!("module `{}`", tcx.def_path_str(def_id))
    }
}
