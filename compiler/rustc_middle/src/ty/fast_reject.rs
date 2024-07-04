use rustc_hir::def_id::DefId;

use super::TyCtxt;

pub use rustc_type_ir::fast_reject::*;

pub type DeepRejectCtxt<'tcx> = rustc_type_ir::fast_reject::DeepRejectCtxt<TyCtxt<'tcx>>;

pub type SimplifiedType = rustc_type_ir::fast_reject::SimplifiedType<DefId>;
