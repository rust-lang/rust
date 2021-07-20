//! This file provides API for compiler consumers.

use rustc_hir::def_id::LocalDefId;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::Body;
use rustc_middle::ty::{self, TyCtxt};

pub use super::{
    facts::{AllFacts as PoloniusInput, RustcFacts},
    location::{LocationTable, RichLocation},
    nll::PoloniusOutput,
    BodyWithBorrowckFacts,
};

/// This function computes Polonius facts for the given body. It makes a copy of
/// the body because it needs to regenerate the region identifiers.
pub fn get_body_with_borrowck_facts<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> BodyWithBorrowckFacts<'tcx> {
    let (input_body, promoted) = tcx.mir_promoted(def);
    tcx.infer_ctxt().enter(|infcx| {
        let input_body: &Body<'_> = &input_body.borrow();
        let promoted: &IndexVec<_, _> = &promoted.borrow();
        *super::do_mir_borrowck(&infcx, input_body, promoted, true).1.unwrap()
    })
}
