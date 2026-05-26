use rustc_hir::def_id::DefId;
pub use rustc_type_ir::fast_reject::*;

use super::TyCtxt;

pub type DeepRejectCtxt<'tcx, const HANDLE_LHS: u8, const HANDLE_RHS: u8> =
    rustc_type_ir::fast_reject::DeepRejectCtxt<TyCtxt<'tcx>, HANDLE_LHS, HANDLE_RHS>;

pub type SimplifiedType = rustc_type_ir::fast_reject::SimplifiedType<DefId>;
