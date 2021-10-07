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
/// the body because it needs to regenerate the region identifiers. This function
/// should never be invoked during a typical compilation session due to performance
/// issues with Polonius.
///
/// Note:
/// *   This function will panic if the required body was already stolen. This
///     can, for example, happen when requesting a body of a `const` function
///     because they are evaluated during typechecking. The panic can be avoided
///     by overriding the `mir_borrowck` query. You can find a complete example
///     that shows how to do this at `src/test/run-make/obtain-borrowck/`.
///
/// *   Polonius is highly unstable, so expect regular changes in its signature or other details.
pub fn get_body_with_borrowck_facts<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> BodyWithBorrowckFacts<'tcx> {
    let (input_body, promoted) = tcx.mir_promoted(def);
    tcx.infer_ctxt().with_opaque_type_inference(def.did).enter(|infcx| {
        let input_body: &Body<'_> = &input_body.borrow();
        let promoted: &IndexVec<_, _> = &promoted.borrow();
        *super::do_mir_borrowck(&infcx, input_body, promoted, true).1.unwrap()
    })
}
