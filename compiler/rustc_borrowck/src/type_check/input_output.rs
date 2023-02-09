//! This module contains code to equate the input/output types appearing
//! in the MIR with the expected input/output types from the function
//! signature. This requires a bit of processing, as the expected types
//! are supplied to us before normalization and may contain opaque
//! `impl Trait` instances. In contrast, the input/output types found in
//! the MIR (specifically, in the special local variables for the
//! `RETURN_PLACE` the MIR arguments) are always fully normalized (and
//! contain revealed `impl Trait` values).

use rustc_index::vec::Idx;
use rustc_infer::infer::LateBoundRegionConversionTime;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;

use crate::universal_regions::UniversalRegions;

use super::{Locations, TypeChecker};

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    /// Check explicit closure signature annotation,
    /// e.g., `|x: FxHashMap<_, &'static u32>| ...`.
    #[instrument(skip(self, body), level = "debug")]
    pub(super) fn check_signature_annotation(&mut self, body: &Body<'tcx>) {
        let mir_def_id = body.source.def_id().expect_local();
        if !self.tcx().is_closure(mir_def_id.to_def_id()) {
            return;
        }
        let Some(user_provided_poly_sig) =
            self.tcx().typeck(mir_def_id).user_provided_sigs.get(&mir_def_id)
        else {
            return;
        };

        // Instantiate the canonicalized variables from user-provided signature
        // (e.g., the `_` in the code above) with fresh variables.
        // Then replace the bound items in the fn sig with fresh variables,
        // so that they represent the view from "inside" the closure.
        let user_provided_sig = self
            .instantiate_canonical_with_fresh_inference_vars(body.span, &user_provided_poly_sig);
        let user_provided_sig = self.infcx.instantiate_binder_with_fresh_vars(
            body.span,
            LateBoundRegionConversionTime::FnCall,
            user_provided_sig,
        );

        for (&user_ty, arg_decl) in user_provided_sig.inputs().iter().zip(
            // In MIR, closure args begin with an implicit `self`. Skip it!
            body.args_iter().skip(1).map(|local| &body.local_decls[local]),
        ) {
            self.ascribe_user_type_skip_wf(
                arg_decl.ty,
                ty::UserType::Ty(user_ty),
                arg_decl.source_info.span,
            );
        }

        // If the user explicitly annotated the output type, enforce it.
        let output_decl = &body.local_decls[RETURN_PLACE];
        self.ascribe_user_type_skip_wf(
            output_decl.ty,
            ty::UserType::Ty(user_provided_sig.output()),
            output_decl.source_info.span,
        );
    }

    #[instrument(skip(self, body, universal_regions), level = "debug")]
    pub(super) fn equate_inputs_and_outputs(
        &mut self,
        body: &Body<'tcx>,
        universal_regions: &UniversalRegions<'tcx>,
        normalized_inputs_and_output: &[Ty<'tcx>],
    ) {
        let (&normalized_output_ty, normalized_input_tys) =
            normalized_inputs_and_output.split_last().unwrap();

        debug!(?normalized_output_ty);
        debug!(?normalized_input_tys);

        // Equate expected input tys with those in the MIR.
        for (argument_index, &normalized_input_ty) in normalized_input_tys.iter().enumerate() {
            if argument_index + 1 >= body.local_decls.len() {
                self.tcx()
                    .sess
                    .delay_span_bug(body.span, "found more normalized_input_ty than local_decls");
                break;
            }

            // In MIR, argument N is stored in local N+1.
            let local = Local::new(argument_index + 1);

            let mir_input_ty = body.local_decls[local].ty;

            let mir_input_span = body.local_decls[local].source_info.span;
            self.equate_normalized_input_or_output(
                normalized_input_ty,
                mir_input_ty,
                mir_input_span,
            );
        }

        debug!(
            "equate_inputs_and_outputs: body.yield_ty {:?}, universal_regions.yield_ty {:?}",
            body.yield_ty(),
            universal_regions.yield_ty
        );

        // We will not have a universal_regions.yield_ty if we yield (by accident)
        // outside of a generator and return an `impl Trait`, so emit a delay_span_bug
        // because we don't want to panic in an assert here if we've already got errors.
        if body.yield_ty().is_some() != universal_regions.yield_ty.is_some() {
            self.tcx().sess.delay_span_bug(
                body.span,
                &format!(
                    "Expected body to have yield_ty ({:?}) iff we have a UR yield_ty ({:?})",
                    body.yield_ty(),
                    universal_regions.yield_ty,
                ),
            );
        }

        if let (Some(mir_yield_ty), Some(ur_yield_ty)) =
            (body.yield_ty(), universal_regions.yield_ty)
        {
            let yield_span = body.local_decls[RETURN_PLACE].source_info.span;
            self.equate_normalized_input_or_output(ur_yield_ty, mir_yield_ty, yield_span);
        }

        // Return types are a bit more complex. They may contain opaque `impl Trait` types.
        let mir_output_ty = body.local_decls[RETURN_PLACE].ty;
        let output_span = body.local_decls[RETURN_PLACE].source_info.span;
        if let Err(terr) = self.eq_types(
            normalized_output_ty,
            mir_output_ty,
            Locations::All(output_span),
            ConstraintCategory::BoringNoLocation,
        ) {
            span_mirbug!(
                self,
                Location::START,
                "equate_inputs_and_outputs: `{:?}=={:?}` failed with `{:?}`",
                normalized_output_ty,
                mir_output_ty,
                terr
            );
        };
    }

    #[instrument(skip(self), level = "debug")]
    fn equate_normalized_input_or_output(&mut self, a: Ty<'tcx>, b: Ty<'tcx>, span: Span) {
        if let Err(_) =
            self.eq_types(a, b, Locations::All(span), ConstraintCategory::BoringNoLocation)
        {
            // FIXME(jackh726): This is a hack. It's somewhat like
            // `rustc_traits::normalize_after_erasing_regions`. Ideally, we'd
            // like to normalize *before* inserting into `local_decls`, but
            // doing so ends up causing some other trouble.
            let b = self.normalize(b, Locations::All(span));

            // Note: if we have to introduce new placeholders during normalization above, then we won't have
            // added those universes to the universe info, which we would want in `relate_tys`.
            if let Err(terr) =
                self.eq_types(a, b, Locations::All(span), ConstraintCategory::BoringNoLocation)
            {
                span_mirbug!(
                    self,
                    Location::START,
                    "equate_normalized_input_or_output: `{:?}=={:?}` failed with `{:?}`",
                    a,
                    b,
                    terr
                );
            }
        }
    }
}
