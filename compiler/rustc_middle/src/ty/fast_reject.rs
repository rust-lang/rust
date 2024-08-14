use rustc_hir::def_id::DefId;
pub use rustc_type_ir::fast_reject::*;
pub use rustc_type_ir::new_reject_ctxt;

pub type DeepRejectCtxt<I, const LHS: bool, const RHS: bool> =
    rustc_type_ir::fast_reject::DeepRejectCtxt<I, LHS, RHS>;

pub type SimplifiedType = rustc_type_ir::fast_reject::SimplifiedType<DefId>;
