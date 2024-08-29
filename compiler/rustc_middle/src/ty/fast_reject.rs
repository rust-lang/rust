use rustc_hir::def_id::DefId;
pub use rustc_type_ir::fast_reject::*;

use super::TyCtxt;

pub type DeepRejectCtxt<'tcx> = rustc_type_ir::fast_reject::DeepRejectCtxt<TyCtxt<'tcx>>;

pub type SimplifiedType = rustc_type_ir::fast_reject::SimplifiedType<DefId>;
