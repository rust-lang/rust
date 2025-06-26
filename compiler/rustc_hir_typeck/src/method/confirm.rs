use std::ops::Deref;

use rustc_hir as hir;
use rustc_hir::GenericArg;
use rustc_hir::def_id::DefId;
use rustc_hir_analysis::hir_ty_lowering::generics::{
    check_generic_arg_count_for_call, lower_generic_args,
};
use rustc_hir_analysis::hir_ty_lowering::{
    FeedConstTy, GenericArgsLowerer, HirTyLowerer, IsMethodCall, RegionInferReason,
};
use rustc_infer::infer::{
    BoundRegionConversionTime, DefineOpaqueTypes, InferOk, RegionVariableOrigin,
};
use rustc_lint::builtin::SUPERTRAIT_ITEM_SHADOWING_USAGE;
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability, PointerCoercion,
};
use rustc_middle::ty::{
    self, GenericArgs, GenericArgsRef, GenericParamDefKind, Ty, TyCtxt, TypeFoldable,
    TypeVisitableExt, UserArgs,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{DUMMY_SP, Span};
use rustc_trait_selection::traits;
use tracing::debug;

use super::{MethodCallee, probe};
use crate::errors::{SupertraitItemShadowee, SupertraitItemShadower, SupertraitItemShadowing};
use crate::{FnCtxt, callee};

struct ConfirmContext<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    self_expr: &'tcx hir::Expr<'tcx>,
    call_expr: &'tcx hir::Expr<'tcx>,
    skip_record_for_diagnostics: bool,
}

impl<'a, 'tcx> Deref for ConfirmContext<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        self.fcx
    }
}

#[derive(Debug)]
pub(crate) struct ConfirmResult<'tcx> {
    pub callee: MethodCallee<'tcx>,
    pub illegal_sized_bound: Option<Span>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn confirm_method(
        &self,
        span: Span,
        self_expr: &'tcx hir::Expr<'tcx>,
        call_expr: &'tcx hir::Expr<'tcx>,
        unadjusted_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
        segment: &'tcx hir::PathSegment<'tcx>,
    ) -> ConfirmResult<'tcx> {
        debug!(
            "confirm(unadjusted_self_ty={:?}, pick={:?}, generic_args={:?})",
            unadjusted_self_ty, pick, segment.args,
        );

        let mut confirm_cx = ConfirmContext::new(self, span, self_expr, call_expr);
        confirm_cx.confirm(unadjusted_self_ty, pick, segment)
    }

    pub(crate) fn confirm_method_for_diagnostic(
        &self,
        span: Span,
        self_expr: &'tcx hir::Expr<'tcx>,
        call_expr: &'tcx hir::Expr<'tcx>,
        unadjusted_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
        segment: &hir::PathSegment<'tcx>,
    ) -> ConfirmResult<'tcx> {
        let mut confirm_cx = ConfirmContext::new(self, span, self_expr, call_expr);
        confirm_cx.skip_record_for_diagnostics = true;
        confirm_cx.confirm(unadjusted_self_ty, pick, segment)
    }
}

impl<'a, 'tcx> ConfirmContext<'a, 'tcx> {
    fn new(
        fcx: &'a FnCtxt<'a, 'tcx>,
        span: Span,
        self_expr: &'tcx hir::Expr<'tcx>,
        call_expr: &'tcx hir::Expr<'tcx>,
    ) -> ConfirmContext<'a, 'tcx> {
        ConfirmContext { fcx, span, self_expr, call_expr, skip_record_for_diagnostics: false }
    }

    fn confirm(
        &mut self,
        unadjusted_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
        segment: &hir::PathSegment<'tcx>,
    ) -> ConfirmResult<'tcx> {
        // Adjust the self expression the user provided and obtain the adjusted type.
        let self_ty = self.adjust_self_ty(unadjusted_self_ty, pick);

        // Create generic args for the method's type parameters.
        let rcvr_args = self.fresh_receiver_args(self_ty, pick);
        let all_args = self.instantiate_method_args(pick, segment, rcvr_args);

        debug!("rcvr_args={rcvr_args:?}, all_args={all_args:?}");

        // Create the final signature for the method, replacing late-bound regions.
        let (method_sig, method_predicates) = self.instantiate_method_sig(pick, all_args);

        // If there is a `Self: Sized` bound and `Self` is a trait object, it is possible that
        // something which derefs to `Self` actually implements the trait and the caller
        // wanted to make a static dispatch on it but forgot to import the trait.
        // See test `tests/ui/issue-35976.rs`.
        //
        // In that case, we'll error anyway, but we'll also re-run the search with all traits
        // in scope, and if we find another method which can be used, we'll output an
        // appropriate hint suggesting to import the trait.
        let filler_args = rcvr_args
            .extend_to(self.tcx, pick.item.def_id, |def, _| self.tcx.mk_param_from_def(def));
        let illegal_sized_bound = self.predicates_require_illegal_sized_bound(
            self.tcx.predicates_of(pick.item.def_id).instantiate(self.tcx, filler_args),
        );

        // Unify the (adjusted) self type with what the method expects.
        //
        // SUBTLE: if we want good error messages, because of "guessing" while matching
        // traits, no trait system method can be called before this point because they
        // could alter our Self-type, except for normalizing the receiver from the
        // signature (which is also done during probing).
        let method_sig_rcvr = self.normalize(self.span, method_sig.inputs()[0]);
        debug!(
            "confirm: self_ty={:?} method_sig_rcvr={:?} method_sig={:?} method_predicates={:?}",
            self_ty, method_sig_rcvr, method_sig, method_predicates
        );
        self.unify_receivers(self_ty, method_sig_rcvr, pick);

        let (method_sig, method_predicates) =
            self.normalize(self.span, (method_sig, method_predicates));
        let method_sig = ty::Binder::dummy(method_sig);

        // Make sure nobody calls `drop()` explicitly.
        self.check_for_illegal_method_calls(pick);

        // Lint when an item is shadowing a supertrait item.
        self.lint_shadowed_supertrait_items(pick, segment);

        // Add any trait/regions obligations specified on the method's type parameters.
        // We won't add these if we encountered an illegal sized bound, so that we can use
        // a custom error in that case.
        if illegal_sized_bound.is_none() {
            self.add_obligations(
                Ty::new_fn_ptr(self.tcx, method_sig),
                all_args,
                method_predicates,
                pick.item.def_id,
            );
        }

        // Create the final `MethodCallee`.
        let callee = MethodCallee {
            def_id: pick.item.def_id,
            args: all_args,
            sig: method_sig.skip_binder(),
        };
        ConfirmResult { callee, illegal_sized_bound }
    }

    ///////////////////////////////////////////////////////////////////////////
    // ADJUSTMENTS

    fn adjust_self_ty(
        &mut self,
        unadjusted_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
    ) -> Ty<'tcx> {
        // Commit the autoderefs by calling `autoderef` again, but this
        // time writing the results into the various typeck results.
        let mut autoderef = self.autoderef(self.call_expr.span, unadjusted_self_ty);
        let Some((ty, n)) = autoderef.nth(pick.autoderefs) else {
            return Ty::new_error_with_message(
                self.tcx,
                DUMMY_SP,
                format!("failed autoderef {}", pick.autoderefs),
            );
        };
        assert_eq!(n, pick.autoderefs);

        let mut adjustments = self.adjust_steps(&autoderef);
        let mut target = self.structurally_resolve_type(autoderef.span(), ty);

        match pick.autoref_or_ptr_adjustment {
            Some(probe::AutorefOrPtrAdjustment::Autoref { mutbl, unsize }) => {
                let region = self.next_region_var(RegionVariableOrigin::Autoref(self.span));
                // Type we're wrapping in a reference, used later for unsizing
                let base_ty = target;

                target = Ty::new_ref(self.tcx, region, target, mutbl);

                // Method call receivers are the primary use case
                // for two-phase borrows.
                let mutbl = AutoBorrowMutability::new(mutbl, AllowTwoPhase::Yes);

                adjustments
                    .push(Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)), target });

                if unsize {
                    let unsized_ty = if let ty::Array(elem_ty, _) = base_ty.kind() {
                        Ty::new_slice(self.tcx, *elem_ty)
                    } else {
                        bug!(
                            "AutorefOrPtrAdjustment's unsize flag should only be set for array ty, found {}",
                            base_ty
                        )
                    };
                    target = Ty::new_ref(self.tcx, region, unsized_ty, mutbl.into());
                    adjustments.push(Adjustment {
                        kind: Adjust::Pointer(PointerCoercion::Unsize),
                        target,
                    });
                }
            }
            Some(probe::AutorefOrPtrAdjustment::ToConstPtr) => {
                target = match target.kind() {
                    &ty::RawPtr(ty, mutbl) => {
                        assert!(mutbl.is_mut());
                        Ty::new_imm_ptr(self.tcx, ty)
                    }
                    other => panic!("Cannot adjust receiver type {other:?} to const ptr"),
                };

                adjustments.push(Adjustment {
                    kind: Adjust::Pointer(PointerCoercion::MutToConstPointer),
                    target,
                });
            }

            Some(probe::AutorefOrPtrAdjustment::ReborrowPin(mutbl)) => {
                let region = self.next_region_var(RegionVariableOrigin::Autoref(self.span));

                target = match target.kind() {
                    ty::Adt(pin, args) if self.tcx.is_lang_item(pin.did(), hir::LangItem::Pin) => {
                        let inner_ty = match args[0].expect_ty().kind() {
                            ty::Ref(_, ty, _) => *ty,
                            _ => bug!("Expected a reference type for argument to Pin"),
                        };
                        Ty::new_pinned_ref(self.tcx, region, inner_ty, mutbl)
                    }
                    _ => bug!("Cannot adjust receiver type for reborrowing pin of {target:?}"),
                };

                adjustments.push(Adjustment { kind: Adjust::ReborrowPin(mutbl), target });
            }
            None => {}
        }

        self.register_predicates(autoderef.into_obligations());

        // Write out the final adjustments.
        if !self.skip_record_for_diagnostics {
            self.apply_adjustments(self.self_expr, adjustments);
        }

        target
    }

    /// Returns a set of generic parameters for the method *receiver* where all type and region
    /// parameters are instantiated with fresh variables. This generic parameters does not include any
    /// parameters declared on the method itself.
    ///
    /// Note that this generic parameters may include late-bound regions from the impl level. If so,
    /// these are instantiated later in the `instantiate_method_sig` routine.
    fn fresh_receiver_args(
        &mut self,
        self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
    ) -> GenericArgsRef<'tcx> {
        match pick.kind {
            probe::InherentImplPick => {
                let impl_def_id = pick.item.container_id(self.tcx);
                assert!(
                    self.tcx.impl_trait_ref(impl_def_id).is_none(),
                    "impl {impl_def_id:?} is not an inherent impl"
                );
                self.fresh_args_for_item(self.span, impl_def_id)
            }

            probe::ObjectPick => {
                let trait_def_id = pick.item.container_id(self.tcx);

                // If the trait is not object safe (specifically, we care about when
                // the receiver is not valid), then there's a chance that we will not
                // actually be able to recover the object by derefing the receiver like
                // we should if it were valid.
                if !self.tcx.is_dyn_compatible(trait_def_id) {
                    return ty::GenericArgs::extend_with_error(self.tcx, trait_def_id, &[]);
                }

                // This shouldn't happen for non-region error kinds, but may occur
                // when we have error regions. Specifically, since we canonicalize
                // during method steps, we may successfully deref when we assemble
                // the pick, but fail to deref when we try to extract the object
                // type from the pick during confirmation. This is fine, we're basically
                // already doomed by this point.
                if self_ty.references_error() {
                    return ty::GenericArgs::extend_with_error(self.tcx, trait_def_id, &[]);
                }

                self.extract_existential_trait_ref(self_ty, |this, object_ty, principal| {
                    // The object data has no entry for the Self
                    // Type. For the purposes of this method call, we
                    // instantiate the object type itself. This
                    // wouldn't be a sound instantiation in all cases,
                    // since each instance of the object type is a
                    // different existential and hence could match
                    // distinct types (e.g., if `Self` appeared as an
                    // argument type), but those cases have already
                    // been ruled out when we deemed the trait to be
                    // "dyn-compatible".
                    let original_poly_trait_ref = principal.with_self_ty(this.tcx, object_ty);
                    let upcast_poly_trait_ref = this.upcast(original_poly_trait_ref, trait_def_id);
                    let upcast_trait_ref =
                        this.instantiate_binder_with_fresh_vars(upcast_poly_trait_ref);
                    debug!(
                        "original_poly_trait_ref={:?} upcast_trait_ref={:?} target_trait={:?}",
                        original_poly_trait_ref, upcast_trait_ref, trait_def_id
                    );
                    upcast_trait_ref.args
                })
            }

            probe::TraitPick => {
                let trait_def_id = pick.item.container_id(self.tcx);

                // Make a trait reference `$0 : Trait<$1...$n>`
                // consisting entirely of type variables. Later on in
                // the process we will unify the transformed-self-type
                // of the method with the actual type in order to
                // unify some of these variables.
                self.fresh_args_for_item(self.span, trait_def_id)
            }

            probe::WhereClausePick(poly_trait_ref) => {
                // Where clauses can have bound regions in them. We need to instantiate
                // those to convert from a poly-trait-ref to a trait-ref.
                self.instantiate_binder_with_fresh_vars(poly_trait_ref).args
            }
        }
    }

    fn extract_existential_trait_ref<R, F>(&mut self, self_ty: Ty<'tcx>, mut closure: F) -> R
    where
        F: FnMut(&mut ConfirmContext<'a, 'tcx>, Ty<'tcx>, ty::PolyExistentialTraitRef<'tcx>) -> R,
    {
        // If we specified that this is an object method, then the
        // self-type ought to be something that can be dereferenced to
        // yield an object-type (e.g., `&Object` or `Box<Object>`
        // etc).

        let mut autoderef = self.fcx.autoderef(self.span, self_ty);

        // We don't need to gate this behind arbitrary self types
        // per se, but it does make things a bit more gated.
        if self.tcx.features().arbitrary_self_types()
            || self.tcx.features().arbitrary_self_types_pointers()
        {
            autoderef = autoderef.use_receiver_trait();
        }

        autoderef
            .include_raw_pointers()
            .find_map(|(ty, _)| match ty.kind() {
                ty::Dynamic(data, ..) => Some(closure(
                    self,
                    ty,
                    data.principal().unwrap_or_else(|| {
                        span_bug!(self.span, "calling trait method on empty object?")
                    }),
                )),
                _ => None,
            })
            .unwrap_or_else(|| {
                span_bug!(
                    self.span,
                    "self-type `{}` for ObjectPick never dereferenced to an object",
                    self_ty
                )
            })
    }

    fn instantiate_method_args(
        &mut self,
        pick: &probe::Pick<'tcx>,
        seg: &hir::PathSegment<'tcx>,
        parent_args: GenericArgsRef<'tcx>,
    ) -> GenericArgsRef<'tcx> {
        // Determine the values for the generic parameters of the method.
        // If they were not explicitly supplied, just construct fresh
        // variables.
        let generics = self.tcx.generics_of(pick.item.def_id);

        let arg_count_correct = check_generic_arg_count_for_call(
            self.fcx,
            pick.item.def_id,
            generics,
            seg,
            IsMethodCall::Yes,
        );

        // Create generic parameters for early-bound lifetime parameters,
        // combining parameters from the type and those from the method.
        assert_eq!(generics.parent_count, parent_args.len());

        struct GenericArgsCtxt<'a, 'tcx> {
            cfcx: &'a ConfirmContext<'a, 'tcx>,
            pick: &'a probe::Pick<'tcx>,
            seg: &'a hir::PathSegment<'tcx>,
        }
        impl<'a, 'tcx> GenericArgsLowerer<'a, 'tcx> for GenericArgsCtxt<'a, 'tcx> {
            fn args_for_def_id(
                &mut self,
                def_id: DefId,
            ) -> (Option<&'a hir::GenericArgs<'tcx>>, bool) {
                if def_id == self.pick.item.def_id {
                    if let Some(data) = self.seg.args {
                        return (Some(data), false);
                    }
                }
                (None, false)
            }

            fn provided_kind(
                &mut self,
                preceding_args: &[ty::GenericArg<'tcx>],
                param: &ty::GenericParamDef,
                arg: &GenericArg<'tcx>,
            ) -> ty::GenericArg<'tcx> {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => self
                        .cfcx
                        .fcx
                        .lowerer()
                        .lower_lifetime(lt, RegionInferReason::Param(param))
                        .into(),
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        // We handle the ambig portions of `Ty` in the match arms below
                        self.cfcx.lower_ty(ty.as_unambig_ty()).raw.into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Infer(inf)) => {
                        self.cfcx.lower_ty(&inf.to_ty()).raw.into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(ct)) => self
                        .cfcx
                        // We handle the ambig portions of `ConstArg` in the match arms below
                        .lower_const_arg(
                            ct.as_unambig_ct(),
                            FeedConstTy::Param(param.def_id, preceding_args),
                        )
                        .into(),
                    (GenericParamDefKind::Const { .. }, GenericArg::Infer(inf)) => {
                        self.cfcx.ct_infer(Some(param), inf.span).into()
                    }
                    (kind, arg) => {
                        bug!("mismatched method arg kind {kind:?} in turbofish: {arg:?}")
                    }
                }
            }

            fn inferred_kind(
                &mut self,
                _preceding_args: &[ty::GenericArg<'tcx>],
                param: &ty::GenericParamDef,
                _infer_args: bool,
            ) -> ty::GenericArg<'tcx> {
                self.cfcx.var_for_def(self.cfcx.span, param)
            }
        }

        let args = lower_generic_args(
            self.fcx,
            pick.item.def_id,
            parent_args,
            false,
            None,
            &arg_count_correct,
            &mut GenericArgsCtxt { cfcx: self, pick, seg },
        );

        // When the method is confirmed, the `args` includes
        // parameters from not just the method, but also the impl of
        // the method -- in particular, the `Self` type will be fully
        // resolved. However, those are not something that the "user
        // specified" -- i.e., those types come from the inferred type
        // of the receiver, not something the user wrote. So when we
        // create the user-args, we want to replace those earlier
        // types with just the types that the user actually wrote --
        // that is, those that appear on the *method itself*.
        //
        // As an example, if the user wrote something like
        // `foo.bar::<u32>(...)` -- the `Self` type here will be the
        // type of `foo` (possibly adjusted), but we don't want to
        // include that. We want just the `[_, u32]` part.
        if !args.is_empty() && !generics.is_own_empty() {
            let user_type_annotation = self.probe(|_| {
                let user_args = UserArgs {
                    args: GenericArgs::for_item(self.tcx, pick.item.def_id, |param, _| {
                        let i = param.index as usize;
                        if i < generics.parent_count {
                            self.fcx.var_for_def(DUMMY_SP, param)
                        } else {
                            args[i]
                        }
                    }),
                    user_self_ty: None, // not relevant here
                };

                self.fcx.canonicalize_user_type_annotation(ty::UserType::new(
                    ty::UserTypeKind::TypeOf(pick.item.def_id, user_args),
                ))
            });

            debug!("instantiate_method_args: user_type_annotation={:?}", user_type_annotation);

            if !self.skip_record_for_diagnostics {
                self.fcx.write_user_type_annotation(self.call_expr.hir_id, user_type_annotation);
            }
        }

        self.normalize(self.span, args)
    }

    fn unify_receivers(
        &mut self,
        self_ty: Ty<'tcx>,
        method_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
    ) {
        debug!(
            "unify_receivers: self_ty={:?} method_self_ty={:?} span={:?} pick={:?}",
            self_ty, method_self_ty, self.span, pick
        );
        let cause = self.cause(self.self_expr.span, ObligationCauseCode::Misc);
        match self.at(&cause, self.param_env).sup(DefineOpaqueTypes::Yes, method_self_ty, self_ty) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
            }
            Err(terr) => {
                if self.tcx.features().arbitrary_self_types() {
                    self.err_ctxt()
                        .report_mismatched_types(
                            &cause,
                            self.param_env,
                            method_self_ty,
                            self_ty,
                            terr,
                        )
                        .emit();
                } else {
                    // This has/will have errored in wfcheck, which we cannot depend on from here, as typeck on functions
                    // may run before wfcheck if the function is used in const eval.
                    self.dcx().span_delayed_bug(
                        cause.span,
                        format!("{self_ty} was a subtype of {method_self_ty} but now is not?"),
                    );
                }
            }
        }
    }

    // NOTE: this returns the *unnormalized* predicates and method sig. Because of
    // inference guessing, the predicates and method signature can't be normalized
    // until we unify the `Self` type.
    fn instantiate_method_sig(
        &mut self,
        pick: &probe::Pick<'tcx>,
        all_args: GenericArgsRef<'tcx>,
    ) -> (ty::FnSig<'tcx>, ty::InstantiatedPredicates<'tcx>) {
        debug!("instantiate_method_sig(pick={:?}, all_args={:?})", pick, all_args);

        // Instantiate the bounds on the method with the
        // type/early-bound-regions instantiations performed. There can
        // be no late-bound regions appearing here.
        let def_id = pick.item.def_id;
        let method_predicates = self.tcx.predicates_of(def_id).instantiate(self.tcx, all_args);

        debug!("method_predicates after instantitation = {:?}", method_predicates);

        let sig = self.tcx.fn_sig(def_id).instantiate(self.tcx, all_args);
        debug!("type scheme instantiated, sig={:?}", sig);

        let sig = self.instantiate_binder_with_fresh_vars(sig);
        debug!("late-bound lifetimes from method instantiated, sig={:?}", sig);

        (sig, method_predicates)
    }

    fn add_obligations(
        &mut self,
        fty: Ty<'tcx>,
        all_args: GenericArgsRef<'tcx>,
        method_predicates: ty::InstantiatedPredicates<'tcx>,
        def_id: DefId,
    ) {
        debug!(
            "add_obligations: fty={:?} all_args={:?} method_predicates={:?} def_id={:?}",
            fty, all_args, method_predicates, def_id
        );

        // FIXME: could replace with the following, but we already calculated `method_predicates`,
        // so we just call `predicates_for_generics` directly to avoid redoing work.
        // `self.add_required_obligations(self.span, def_id, &all_args);`
        for obligation in traits::predicates_for_generics(
            |idx, span| {
                let code = ObligationCauseCode::WhereClauseInExpr(
                    def_id,
                    span,
                    self.call_expr.hir_id,
                    idx,
                );
                self.cause(self.span, code)
            },
            self.param_env,
            method_predicates,
        ) {
            self.register_predicate(obligation);
        }

        // this is a projection from a trait reference, so we have to
        // make sure that the trait reference inputs are well-formed.
        self.add_wf_bounds(all_args, self.call_expr.span);

        // the function type must also be well-formed (this is not
        // implied by the args being well-formed because of inherent
        // impls and late-bound regions - see issue #28609).
        self.register_wf_obligation(fty.into(), self.span, ObligationCauseCode::WellFormed(None));
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY

    fn predicates_require_illegal_sized_bound(
        &self,
        predicates: ty::InstantiatedPredicates<'tcx>,
    ) -> Option<Span> {
        let sized_def_id = self.tcx.lang_items().sized_trait()?;

        traits::elaborate(self.tcx, predicates.predicates.iter().copied())
            // We don't care about regions here.
            .filter_map(|pred| match pred.kind().skip_binder() {
                ty::ClauseKind::Trait(trait_pred) if trait_pred.def_id() == sized_def_id => {
                    let span = predicates
                        .iter()
                        .find_map(|(p, span)| if p == pred { Some(span) } else { None })
                        .unwrap_or(DUMMY_SP);
                    Some((trait_pred, span))
                }
                _ => None,
            })
            .find_map(|(trait_pred, span)| match trait_pred.self_ty().kind() {
                ty::Dynamic(..) => Some(span),
                _ => None,
            })
    }

    fn check_for_illegal_method_calls(&self, pick: &probe::Pick<'_>) {
        // Disallow calls to the method `drop` defined in the `Drop` trait.
        if let Some(trait_def_id) = pick.item.trait_container(self.tcx) {
            if let Err(e) = callee::check_legal_trait_for_method_call(
                self.tcx,
                self.span,
                Some(self.self_expr.span),
                self.call_expr.span,
                trait_def_id,
                self.body_id.to_def_id(),
            ) {
                self.set_tainted_by_errors(e);
            }
        }
    }

    fn lint_shadowed_supertrait_items(
        &self,
        pick: &probe::Pick<'_>,
        segment: &hir::PathSegment<'tcx>,
    ) {
        if pick.shadowed_candidates.is_empty() {
            return;
        }

        let shadower_span = self.tcx.def_span(pick.item.def_id);
        let subtrait = self.tcx.item_name(pick.item.trait_container(self.tcx).unwrap());
        let shadower = SupertraitItemShadower { span: shadower_span, subtrait };

        let shadowee = if let [shadowee] = &pick.shadowed_candidates[..] {
            let shadowee_span = self.tcx.def_span(shadowee.def_id);
            let supertrait = self.tcx.item_name(shadowee.trait_container(self.tcx).unwrap());
            SupertraitItemShadowee::Labeled { span: shadowee_span, supertrait }
        } else {
            let (traits, spans): (Vec<_>, Vec<_>) = pick
                .shadowed_candidates
                .iter()
                .map(|item| {
                    (
                        self.tcx.item_name(item.trait_container(self.tcx).unwrap()),
                        self.tcx.def_span(item.def_id),
                    )
                })
                .unzip();
            SupertraitItemShadowee::Several { traits: traits.into(), spans: spans.into() }
        };

        self.tcx.emit_node_span_lint(
            SUPERTRAIT_ITEM_SHADOWING_USAGE,
            segment.hir_id,
            segment.ident.span,
            SupertraitItemShadowing { shadower, shadowee, item: segment.ident.name, subtrait },
        );
    }

    fn upcast(
        &mut self,
        source_trait_ref: ty::PolyTraitRef<'tcx>,
        target_trait_def_id: DefId,
    ) -> ty::PolyTraitRef<'tcx> {
        let upcast_trait_refs =
            traits::upcast_choices(self.tcx, source_trait_ref, target_trait_def_id);

        // must be exactly one trait ref or we'd get an ambig error etc
        if let &[upcast_trait_ref] = upcast_trait_refs.as_slice() {
            upcast_trait_ref
        } else {
            self.dcx().span_delayed_bug(
                self.span,
                format!(
                    "cannot uniquely upcast `{:?}` to `{:?}`: `{:?}`",
                    source_trait_ref, target_trait_def_id, upcast_trait_refs
                ),
            );

            ty::Binder::dummy(ty::TraitRef::new_from_args(
                self.tcx,
                target_trait_def_id,
                ty::GenericArgs::extend_with_error(self.tcx, target_trait_def_id, &[]),
            ))
        }
    }

    fn instantiate_binder_with_fresh_vars<T>(&self, value: ty::Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        self.fcx.instantiate_binder_with_fresh_vars(
            self.span,
            BoundRegionConversionTime::FnCall,
            value,
        )
    }
}
