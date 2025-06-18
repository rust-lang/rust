//! This module contains code to equate the input/output types appearing
//! in the MIR with the expected input/output types from the function
//! signature. This requires a bit of processing, as the expected types
//! are supplied to us before normalization and may contain opaque
//! `impl Trait` instances. In contrast, the input/output types found in
//! the MIR (specifically, in the special local variables for the
//! `RETURN_PLACE` the MIR arguments) are always fully normalized (and
//! contain revealed `impl Trait` values).

use std::assert_matches::assert_matches;

use itertools::Itertools;
use rustc_hir as hir;
use rustc_infer::infer::{BoundRegionConversionTime, RegionVariableOrigin};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use tracing::{debug, instrument};

use super::{Locations, TypeChecker};
use crate::renumber::RegionCtxt;
use crate::universal_regions::DefiningTy;

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    /// Check explicit closure signature annotation,
    /// e.g., `|x: FxIndexMap<_, &'static u32>| ...`.
    #[instrument(skip(self), level = "debug")]
    pub(super) fn check_signature_annotation(&mut self) {
        let mir_def_id = self.body.source.def_id().expect_local();

        if !self.tcx().is_closure_like(mir_def_id.to_def_id()) {
            return;
        }

        let user_provided_poly_sig = self.tcx().closure_user_provided_sig(mir_def_id);

        // Instantiate the canonicalized variables from user-provided signature
        // (e.g., the `_` in the code above) with fresh variables.
        // Then replace the bound items in the fn sig with fresh variables,
        // so that they represent the view from "inside" the closure.
        let user_provided_sig = self.instantiate_canonical(self.body.span, &user_provided_poly_sig);
        let mut user_provided_sig = self.infcx.instantiate_binder_with_fresh_vars(
            self.body.span,
            BoundRegionConversionTime::FnCall,
            user_provided_sig,
        );

        // FIXME(async_closures): It's kind of wacky that we must apply this
        // transformation here, since we do the same thing in HIR typeck.
        // Maybe we could just fix up the canonicalized signature during HIR typeck?
        if let DefiningTy::CoroutineClosure(_, args) = self.universal_regions.defining_ty {
            assert_matches!(
                self.tcx().coroutine_kind(self.tcx().coroutine_for_closure(mir_def_id)),
                Some(hir::CoroutineKind::Desugared(
                    hir::CoroutineDesugaring::Async | hir::CoroutineDesugaring::Gen,
                    hir::CoroutineSource::Closure
                )),
                "this needs to be modified if we're lowering non-async closures"
            );
            // Make sure to use the args from `DefiningTy` so the right NLL region vids are
            // prepopulated into the type.
            let args = args.as_coroutine_closure();
            let tupled_upvars_ty = ty::CoroutineClosureSignature::tupled_upvars_by_closure_kind(
                self.tcx(),
                args.kind(),
                Ty::new_tup(self.tcx(), user_provided_sig.inputs()),
                args.tupled_upvars_ty(),
                args.coroutine_captures_by_ref_ty(),
                self.infcx.next_region_var(RegionVariableOrigin::Misc(self.body.span), || {
                    RegionCtxt::Unknown
                }),
            );

            let next_ty_var = || self.infcx.next_ty_var(self.body.span);
            let output_ty = Ty::new_coroutine(
                self.tcx(),
                self.tcx().coroutine_for_closure(mir_def_id),
                ty::CoroutineArgs::new(
                    self.tcx(),
                    ty::CoroutineArgsParts {
                        parent_args: args.parent_args(),
                        kind_ty: Ty::from_coroutine_closure_kind(self.tcx(), args.kind()),
                        return_ty: user_provided_sig.output(),
                        tupled_upvars_ty,
                        // For async closures, none of these can be annotated, so just fill
                        // them with fresh ty vars.
                        resume_ty: next_ty_var(),
                        yield_ty: next_ty_var(),
                        witness: next_ty_var(),
                    },
                )
                .args,
            );

            user_provided_sig = self.tcx().mk_fn_sig(
                user_provided_sig.inputs().iter().copied(),
                output_ty,
                user_provided_sig.c_variadic,
                user_provided_sig.safety,
                user_provided_sig.abi,
            );
        }

        let is_coroutine_with_implicit_resume_ty = self.tcx().is_coroutine(mir_def_id.to_def_id())
            && user_provided_sig.inputs().is_empty();

        for (&user_ty, arg_decl) in user_provided_sig.inputs().iter().zip_eq(
            // In MIR, closure args begin with an implicit `self`.
            // Also, coroutines have a resume type which may be implicitly `()`.
            self.body
                .args_iter()
                .skip(1 + if is_coroutine_with_implicit_resume_ty { 1 } else { 0 })
                .map(|local| &self.body.local_decls[local]),
        ) {
            self.ascribe_user_type_skip_wf(
                arg_decl.ty,
                ty::UserType::new(ty::UserTypeKind::Ty(user_ty)),
                arg_decl.source_info.span,
            );
        }

        // If the user explicitly annotated the output type, enforce it.
        let output_decl = &self.body.local_decls[RETURN_PLACE];
        self.ascribe_user_type_skip_wf(
            output_decl.ty,
            ty::UserType::new(ty::UserTypeKind::Ty(user_provided_sig.output())),
            output_decl.source_info.span,
        );
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn equate_inputs_and_outputs(&mut self, normalized_inputs_and_output: &[Ty<'tcx>]) {
        let (&normalized_output_ty, normalized_input_tys) =
            normalized_inputs_and_output.split_last().unwrap();

        debug!(?normalized_output_ty);
        debug!(?normalized_input_tys);

        // Equate expected input tys with those in the MIR.
        for (argument_index, &normalized_input_ty) in normalized_input_tys.iter().enumerate() {
            if argument_index + 1 >= self.body.local_decls.len() {
                self.tcx()
                    .dcx()
                    .span_bug(self.body.span, "found more normalized_input_ty than local_decls");
            }

            // In MIR, argument N is stored in local N+1.
            let local = Local::from_usize(argument_index + 1);

            let mir_input_ty = self.body.local_decls[local].ty;

            let mir_input_span = self.body.local_decls[local].source_info.span;
            self.equate_normalized_input_or_output(
                normalized_input_ty,
                mir_input_ty,
                mir_input_span,
            );
        }

        if let Some(mir_yield_ty) = self.body.yield_ty() {
            let yield_span = self.body.local_decls[RETURN_PLACE].source_info.span;
            self.equate_normalized_input_or_output(
                self.universal_regions.yield_ty.unwrap(),
                mir_yield_ty,
                yield_span,
            );
        }

        if let Some(mir_resume_ty) = self.body.resume_ty() {
            let yield_span = self.body.local_decls[RETURN_PLACE].source_info.span;
            self.equate_normalized_input_or_output(
                self.universal_regions.resume_ty.unwrap(),
                mir_resume_ty,
                yield_span,
            );
        }

        // Return types are a bit more complex. They may contain opaque `impl Trait` types.
        let mir_output_ty = self.body.local_decls[RETURN_PLACE].ty;
        let output_span = self.body.local_decls[RETURN_PLACE].source_info.span;
        self.equate_normalized_input_or_output(normalized_output_ty, mir_output_ty, output_span);
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

            // Note: if we have to introduce new placeholders during normalization above, then we
            // won't have added those universes to the universe info, which we would want in
            // `relate_tys`.
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
