use rustc_hir::def_id::DefId;
pub use rustc_type_ir::fast_reject::*;
pub use rustc_type_ir::new_reject_ctxt;

use super::TyCtxt;

pub type DeepRejectCtxt<'tcx, const TREAT_LHS_PARAMS: bool, const TREAT_RHS_PARAMS: bool> =
    rustc_type_ir::fast_reject::DeepRejectCtxt<TyCtxt<'tcx>, TREAT_LHS_PARAMS, TREAT_RHS_PARAMS>;

pub type SimplifiedType = rustc_type_ir::fast_reject::SimplifiedType<DefId>;
