//! This module contains code to equate the input/output types appearing
//! in the MIR with the expected input/output types from the function
//! signature. This requires a bit of processing, as the expected types
//! are supplied to us before normalization and may contain opaque
//! `impl Trait` instances. In contrast, the input/output types found in
//! the MIR (specifically, in the special local variables for the
//! `RETURN_PLACE` the MIR arguments) are always fully normalized (and
//! contain revealed `impl Trait` values).

use crate::type_check::constraint_conversion::ConstraintConversion;
use rustc_index::vec::Idx;
use rustc_infer::infer::LateBoundRegionConversionTime;
use rustc_middle::mir::*;
use rustc_middle::ty::Ty;
use rustc_span::Span;
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits::query::type_op::{self, TypeOp};
use rustc_trait_selection::traits::query::Fallible;
use type_op::TypeOpOutput;

use crate::universal_regions::UniversalRegions;

use super::{Locations, TypeChecker};

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
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

        let mir_def_id = body.source.def_id().expect_local();

        // If the user explicitly annotated the input types, extract
        // those.
        //
        // e.g., `|x: FxHashMap<_, &'static u32>| ...`
        let user_provided_sig;
        if !self.tcx().is_closure(mir_def_id.to_def_id()) {
            user_provided_sig = None;
        } else {
            let typeck_results = self.tcx().typeck(mir_def_id);
            user_provided_sig = typeck_results.user_provided_sigs.get(&mir_def_id.to_def_id()).map(
                |user_provided_poly_sig| {
                    // Instantiate the canonicalized variables from
                    // user-provided signature (e.g., the `_` in the code
                    // above) with fresh variables.
                    let poly_sig = self.instantiate_canonical_with_fresh_inference_vars(
                        body.span,
                        &user_provided_poly_sig,
                    );

                    // Replace the bound items in the fn sig with fresh
                    // variables, so that they represent the view from
                    // "inside" the closure.
                    self.infcx
                        .replace_bound_vars_with_fresh_vars(
                            body.span,
                            LateBoundRegionConversionTime::FnCall,
                            poly_sig,
                        )
                        .0
                },
            );
        }

        debug!(?normalized_input_tys, ?body.local_decls);

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

        if let Some(user_provided_sig) = user_provided_sig {
            for (argument_index, &user_provided_input_ty) in
                user_provided_sig.inputs().iter().enumerate()
            {
                // In MIR, closures begin an implicit `self`, so
                // argument N is stored in local N+2.
                let local = Local::new(argument_index + 2);
                let mir_input_ty = body.local_decls[local].ty;
                let mir_input_span = body.local_decls[local].source_info.span;

                // If the user explicitly annotated the input types, enforce those.
                let user_provided_input_ty =
                    self.normalize(user_provided_input_ty, Locations::All(mir_input_span));

                self.equate_normalized_input_or_output(
                    user_provided_input_ty,
                    mir_input_ty,
                    mir_input_span,
                );
            }
        }

        assert!(body.yield_ty().is_some() == universal_regions.yield_ty.is_some());
        if let Some(mir_yield_ty) = body.yield_ty() {
            let ur_yield_ty = universal_regions.yield_ty.unwrap();
            let yield_span = body.local_decls[RETURN_PLACE].source_info.span;
            self.equate_normalized_input_or_output(ur_yield_ty, mir_yield_ty, yield_span);
        }

        // Return types are a bit more complex. They may contain opaque `impl Trait` types.
        let mir_output_ty = body.local_decls[RETURN_PLACE].ty;
        let output_span = body.local_decls[RETURN_PLACE].source_info.span;
        if let Err(terr) = self.eq_opaque_type_and_type(
            mir_output_ty,
            normalized_output_ty,
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

        // If the user explicitly annotated the output types, enforce those.
        // Note that this only happens for closures.
        if let Some(user_provided_sig) = user_provided_sig {
            let user_provided_output_ty = user_provided_sig.output();
            let user_provided_output_ty =
                self.normalize(user_provided_output_ty, Locations::All(output_span));
            if let Err(err) = self.eq_opaque_type_and_type(
                mir_output_ty,
                user_provided_output_ty,
                Locations::All(output_span),
                ConstraintCategory::BoringNoLocation,
            ) {
                span_mirbug!(
                    self,
                    Location::START,
                    "equate_inputs_and_outputs: `{:?}=={:?}` failed with `{:?}`",
                    mir_output_ty,
                    user_provided_output_ty,
                    err
                );
            }
        }
    }

    #[instrument(skip(self, span), level = "debug")]
    fn equate_normalized_input_or_output(&mut self, a: Ty<'tcx>, b: Ty<'tcx>, span: Span) {
        if let Err(_) =
            self.eq_types(a, b, Locations::All(span), ConstraintCategory::BoringNoLocation)
        {
            // FIXME(jackh726): This is a hack. It's somewhat like
            // `rustc_traits::normalize_after_erasing_regions`. Ideally, we'd
            // like to normalize *before* inserting into `local_decls`, but
            // doing so ends up causing some other trouble.
            let b = match self.normalize_and_add_constraints(b) {
                Ok(n) => n,
                Err(_) => {
                    debug!("equate_inputs_and_outputs: NoSolution");
                    b
                }
            };

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

    pub(crate) fn normalize_and_add_constraints(&mut self, t: Ty<'tcx>) -> Fallible<Ty<'tcx>> {
        let TypeOpOutput { output: norm_ty, constraints, .. } =
            self.param_env.and(type_op::normalize::Normalize::new(t)).fully_perform(self.infcx)?;

        debug!("{:?} normalized to {:?}", t, norm_ty);

        for data in constraints.into_iter().collect::<Vec<_>>() {
            ConstraintConversion::new(
                self.infcx,
                &self.borrowck_context.universal_regions,
                &self.region_bound_pairs,
                Some(self.implicit_region_bound),
                self.param_env,
                Locations::All(DUMMY_SP),
                ConstraintCategory::Internal,
                &mut self.borrowck_context.constraints,
            )
            .convert_all(&*data);
        }

        Ok(norm_ty)
    }
}
