//! This module contains code to equate the input/output types appearing
//! in the MIR with the expected input/output types from the function
//! signature. This requires a bit of processing, as the expected types
//! are supplied to us before normalization and may contain opaque
//! `impl Trait` instances. In contrast, the input/output types found in
//! the MIR (specifically, in the special local variables for the
//! `RETURN_PLACE` the MIR arguments) are always fully normalized (and
//! contain revealed `impl Trait` values).

use itertools::Itertools;
use rustc_data_structures::assert_matches;
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

    //  FIXME(BoxyUwU): This should probably be part of a larger borrowck dev-guide chapter
    //
    /// Enforce that the types of the locals corresponding to the inputs and output of
    /// the body are equal to those of the (normalized) signature.
    ///
    /// This is necessary for two reasons:
    /// - Locals in the MIR all start out with `'erased` regions and then are replaced
    ///    with unconstrained nll vars. If we have a function returning `&'a u32` then
    ///    the local `_0: &'?10 u32` needs to have its region var equated with the nll
    ///    var representing `'a`. i.e. borrow check must uphold that `'?10 = 'a`.
    /// - When computing the normalized signature we may introduce new unconstrained nll
    ///    vars due to higher ranked where clauses ([#136547]). We then wind up with implied
    ///    bounds involving these vars.
    ///
    ///    For this reason it is important that we equate with the *normalized* signature
    ///    which was produced when computing implied bounds. If we do not do so then we will
    ///    wind up with implied bounds on nll vars which cannot actually be used as the nll
    ///    var never gets related to anything.
    ///
    /// For 'closure-like' bodies this function effectively relates the *inferred* signature
    /// of the closure against the locals corresponding to the closure's inputs/output. It *does
    /// not* relate the user provided types for the signature to the locals, this is handled
    /// separately by: [`TypeChecker::check_signature_annotation`].
    ///
    /// [#136547]: <https://www.github.com/rust-lang/rust/issues/136547>
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

        // Equate expected output ty with the type of the RETURN_PLACE in MIR
        let mir_output_ty = self.body.return_ty();
        let output_span = self.body.local_decls[RETURN_PLACE].source_info.span;
        self.equate_normalized_input_or_output(normalized_output_ty, mir_output_ty, output_span);
    }

    #[instrument(skip(self), level = "debug")]
    fn equate_normalized_input_or_output(&mut self, a: Ty<'tcx>, b: Ty<'tcx>, span: Span) {
        if self.infcx.next_trait_solver() {
            return self
                .eq_types(a, b, Locations::All(span), ConstraintCategory::BoringNoLocation)
                .unwrap_or_else(|terr| {
                    span_mirbug!(
                        self,
                        Location::START,
                        "equate_normalized_input_or_output: `{a:?}=={b:?}` failed with `{terr:?}`",
                    );
                });
        }

        // This is a hack. `body.local_decls` are not necessarily normalized in the old
        // solver due to not deeply normalizing in writeback. So we must re-normalize here.
        //
        // However, in most cases normalizing is unnecessary so we only do so if it may be
        // necessary for type equality to hold. This leads to some (very minor) performance
        // wins.
        if let Err(_) =
            self.eq_types(a, b, Locations::All(span), ConstraintCategory::BoringNoLocation)
        {
            let b = self.normalize(b, Locations::All(span));
            self.eq_types(a, b, Locations::All(span), ConstraintCategory::BoringNoLocation)
                .unwrap_or_else(|terr| {
                    span_mirbug!(
                        self,
                        Location::START,
                        "equate_normalized_input_or_output: `{a:?}=={b:?}` failed with `{terr:?}`",
                    );
                });
        };
    }
}
