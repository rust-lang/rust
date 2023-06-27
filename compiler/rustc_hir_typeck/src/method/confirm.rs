use super::{probe, MethodCallee};

use crate::{callee, FnCtxt};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::GenericArg;
use rustc_hir_analysis::astconv::generics::{
    check_generic_arg_count_for_call, create_substs_for_generic_args,
};
use rustc_hir_analysis::astconv::{AstConv, CreateSubstsForGenericArgsCtxt, IsMethodCall};
use rustc_infer::infer::{self, DefineOpaqueTypes, InferOk};
use rustc_middle::traits::{ObligationCauseCode, UnifyReceiverContext};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, PointerCast};
use rustc_middle::ty::adjustment::{AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::{self, SubstsRef};
use rustc_middle::ty::{self, GenericParamDefKind, Ty, TyCtxt};
use rustc_middle::ty::{InternalSubsts, UserSubsts, UserType};
use rustc_span::{Span, DUMMY_SP};
use rustc_trait_selection::traits;

use std::ops::Deref;

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
pub struct ConfirmResult<'tcx> {
    pub callee: MethodCallee<'tcx>,
    pub illegal_sized_bound: Option<Span>,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn confirm_method(
        &self,
        span: Span,
        self_expr: &'tcx hir::Expr<'tcx>,
        call_expr: &'tcx hir::Expr<'tcx>,
        unadjusted_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
        segment: &hir::PathSegment<'_>,
    ) -> ConfirmResult<'tcx> {
        debug!(
            "confirm(unadjusted_self_ty={:?}, pick={:?}, generic_args={:?})",
            unadjusted_self_ty, pick, segment.args,
        );

        let mut confirm_cx = ConfirmContext::new(self, span, self_expr, call_expr);
        confirm_cx.confirm(unadjusted_self_ty, pick, segment)
    }

    pub fn confirm_method_for_diagnostic(
        &self,
        span: Span,
        self_expr: &'tcx hir::Expr<'tcx>,
        call_expr: &'tcx hir::Expr<'tcx>,
        unadjusted_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
        segment: &hir::PathSegment<'_>,
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
        segment: &hir::PathSegment<'_>,
    ) -> ConfirmResult<'tcx> {
        // Adjust the self expression the user provided and obtain the adjusted type.
        let self_ty = self.adjust_self_ty(unadjusted_self_ty, &pick);

        // Create substitutions for the method's type parameters.
        let rcvr_substs = self.fresh_receiver_substs(self_ty, &pick);
        let all_substs = self.instantiate_method_substs(&pick, segment, rcvr_substs);

        debug!("rcvr_substs={rcvr_substs:?}, all_substs={all_substs:?}");

        // Create the final signature for the method, replacing late-bound regions.
        let (method_sig, method_predicates) = self.instantiate_method_sig(&pick, all_substs);

        // If there is a `Self: Sized` bound and `Self` is a trait object, it is possible that
        // something which derefs to `Self` actually implements the trait and the caller
        // wanted to make a static dispatch on it but forgot to import the trait.
        // See test `tests/ui/issue-35976.rs`.
        //
        // In that case, we'll error anyway, but we'll also re-run the search with all traits
        // in scope, and if we find another method which can be used, we'll output an
        // appropriate hint suggesting to import the trait.
        let filler_substs = rcvr_substs
            .extend_to(self.tcx, pick.item.def_id, |def, _| self.tcx.mk_param_from_def(def));
        let illegal_sized_bound = self.predicates_require_illegal_sized_bound(
            self.tcx.predicates_of(pick.item.def_id).instantiate(self.tcx, filler_substs),
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
        self.unify_receivers(self_ty, method_sig_rcvr, &pick, all_substs);

        let (method_sig, method_predicates) =
            self.normalize(self.span, (method_sig, method_predicates));
        let method_sig = ty::Binder::dummy(method_sig);

        // Make sure nobody calls `drop()` explicitly.
        self.enforce_illegal_method_limitations(&pick);

        // Add any trait/regions obligations specified on the method's type parameters.
        // We won't add these if we encountered an illegal sized bound, so that we can use
        // a custom error in that case.
        if illegal_sized_bound.is_none() {
            self.add_obligations(
                self.tcx.mk_fn_ptr(method_sig),
                all_substs,
                method_predicates,
                pick.item.def_id,
            );
        }

        // Create the final `MethodCallee`.
        let callee = MethodCallee {
            def_id: pick.item.def_id,
            substs: all_substs,
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
            return self.tcx.ty_error_with_message(
                rustc_span::DUMMY_SP,
                format!("failed autoderef {}", pick.autoderefs),
            );
        };
        assert_eq!(n, pick.autoderefs);

        let mut adjustments = self.adjust_steps(&autoderef);
        let mut target = self.structurally_resolve_type(autoderef.span(), ty);

        match pick.autoref_or_ptr_adjustment {
            Some(probe::AutorefOrPtrAdjustment::Autoref { mutbl, unsize }) => {
                let region = self.next_region_var(infer::Autoref(self.span));
                // Type we're wrapping in a reference, used later for unsizing
                let base_ty = target;

                target = self.tcx.mk_ref(region, ty::TypeAndMut { mutbl, ty: target });

                // Method call receivers are the primary use case
                // for two-phase borrows.
                let mutbl = AutoBorrowMutability::new(mutbl, AllowTwoPhase::Yes);

                adjustments.push(Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)),
                    target,
                });

                if unsize {
                    let unsized_ty = if let ty::Array(elem_ty, _) = base_ty.kind() {
                        self.tcx.mk_slice(*elem_ty)
                    } else {
                        bug!(
                            "AutorefOrPtrAdjustment's unsize flag should only be set for array ty, found {}",
                            base_ty
                        )
                    };
                    target = self
                        .tcx
                        .mk_ref(region, ty::TypeAndMut { mutbl: mutbl.into(), ty: unsized_ty });
                    adjustments
                        .push(Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), target });
                }
            }
            Some(probe::AutorefOrPtrAdjustment::ToConstPtr) => {
                target = match target.kind() {
                    &ty::RawPtr(ty::TypeAndMut { ty, mutbl }) => {
                        assert!(mutbl.is_mut());
                        self.tcx.mk_ptr(ty::TypeAndMut { mutbl: hir::Mutability::Not, ty })
                    }
                    other => panic!("Cannot adjust receiver type {:?} to const ptr", other),
                };

                adjustments.push(Adjustment {
                    kind: Adjust::Pointer(PointerCast::MutToConstPointer),
                    target,
                });
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

    /// Returns a set of substitutions for the method *receiver* where all type and region
    /// parameters are instantiated with fresh variables. This substitution does not include any
    /// parameters declared on the method itself.
    ///
    /// Note that this substitution may include late-bound regions from the impl level. If so,
    /// these are instantiated later in the `instantiate_method_sig` routine.
    fn fresh_receiver_substs(
        &mut self,
        self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
    ) -> SubstsRef<'tcx> {
        match pick.kind {
            probe::InherentImplPick => {
                let impl_def_id = pick.item.container_id(self.tcx);
                assert!(
                    self.tcx.impl_trait_ref(impl_def_id).is_none(),
                    "impl {:?} is not an inherent impl",
                    impl_def_id
                );
                self.fresh_substs_for_item(self.span, impl_def_id)
            }

            probe::ObjectPick => {
                let trait_def_id = pick.item.container_id(self.tcx);
                self.extract_existential_trait_ref(self_ty, |this, object_ty, principal| {
                    // The object data has no entry for the Self
                    // Type. For the purposes of this method call, we
                    // substitute the object type itself. This
                    // wouldn't be a sound substitution in all cases,
                    // since each instance of the object type is a
                    // different existential and hence could match
                    // distinct types (e.g., if `Self` appeared as an
                    // argument type), but those cases have already
                    // been ruled out when we deemed the trait to be
                    // "object safe".
                    let original_poly_trait_ref = principal.with_self_ty(this.tcx, object_ty);
                    let upcast_poly_trait_ref = this.upcast(original_poly_trait_ref, trait_def_id);
                    let upcast_trait_ref =
                        this.instantiate_binder_with_fresh_vars(upcast_poly_trait_ref);
                    debug!(
                        "original_poly_trait_ref={:?} upcast_trait_ref={:?} target_trait={:?}",
                        original_poly_trait_ref, upcast_trait_ref, trait_def_id
                    );
                    upcast_trait_ref.substs
                })
            }

            probe::TraitPick => {
                let trait_def_id = pick.item.container_id(self.tcx);

                // Make a trait reference `$0 : Trait<$1...$n>`
                // consisting entirely of type variables. Later on in
                // the process we will unify the transformed-self-type
                // of the method with the actual type in order to
                // unify some of these variables.
                self.fresh_substs_for_item(self.span, trait_def_id)
            }

            probe::WhereClausePick(poly_trait_ref) => {
                // Where clauses can have bound regions in them. We need to instantiate
                // those to convert from a poly-trait-ref to a trait-ref.
                self.instantiate_binder_with_fresh_vars(poly_trait_ref).substs
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

        // FIXME: this feels, like, super dubious
        self.fcx
            .autoderef(self.span, self_ty)
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

    fn instantiate_method_substs(
        &mut self,
        pick: &probe::Pick<'tcx>,
        seg: &hir::PathSegment<'_>,
        parent_substs: SubstsRef<'tcx>,
    ) -> SubstsRef<'tcx> {
        // Determine the values for the generic parameters of the method.
        // If they were not explicitly supplied, just construct fresh
        // variables.
        let generics = self.tcx.generics_of(pick.item.def_id);

        let arg_count_correct = check_generic_arg_count_for_call(
            self.tcx,
            self.span,
            pick.item.def_id,
            generics,
            seg,
            IsMethodCall::Yes,
        );

        // Create subst for early-bound lifetime parameters, combining
        // parameters from the type and those from the method.
        assert_eq!(generics.parent_count, parent_substs.len());

        struct MethodSubstsCtxt<'a, 'tcx> {
            cfcx: &'a ConfirmContext<'a, 'tcx>,
            pick: &'a probe::Pick<'tcx>,
            seg: &'a hir::PathSegment<'a>,
        }
        impl<'a, 'tcx> CreateSubstsForGenericArgsCtxt<'a, 'tcx> for MethodSubstsCtxt<'a, 'tcx> {
            fn args_for_def_id(
                &mut self,
                def_id: DefId,
            ) -> (Option<&'a hir::GenericArgs<'a>>, bool) {
                if def_id == self.pick.item.def_id {
                    if let Some(data) = self.seg.args {
                        return (Some(data), false);
                    }
                }
                (None, false)
            }

            fn provided_kind(
                &mut self,
                param: &ty::GenericParamDef,
                arg: &GenericArg<'_>,
            ) -> subst::GenericArg<'tcx> {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        self.cfcx.fcx.astconv().ast_region_to_region(lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.cfcx.to_ty(ty).raw.into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Const(ct)) => {
                        self.cfcx.const_arg_to_const(&ct.value, param.def_id).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Infer(inf)) => {
                        self.cfcx.ty_infer(Some(param), inf.span).into()
                    }
                    (GenericParamDefKind::Const { .. }, GenericArg::Infer(inf)) => {
                        let tcx = self.cfcx.tcx();
                        self.cfcx
                            .ct_infer(
                                tcx.type_of(param.def_id)
                                    .no_bound_vars()
                                    .expect("const parameter types cannot be generic"),
                                Some(param),
                                inf.span,
                            )
                            .into()
                    }
                    _ => unreachable!(),
                }
            }

            fn inferred_kind(
                &mut self,
                _substs: Option<&[subst::GenericArg<'tcx>]>,
                param: &ty::GenericParamDef,
                _infer_args: bool,
            ) -> subst::GenericArg<'tcx> {
                self.cfcx.var_for_def(self.cfcx.span, param)
            }
        }

        let substs = create_substs_for_generic_args(
            self.tcx,
            pick.item.def_id,
            parent_substs,
            false,
            None,
            &arg_count_correct,
            &mut MethodSubstsCtxt { cfcx: self, pick, seg },
        );

        // When the method is confirmed, the `substs` includes
        // parameters from not just the method, but also the impl of
        // the method -- in particular, the `Self` type will be fully
        // resolved. However, those are not something that the "user
        // specified" -- i.e., those types come from the inferred type
        // of the receiver, not something the user wrote. So when we
        // create the user-substs, we want to replace those earlier
        // types with just the types that the user actually wrote --
        // that is, those that appear on the *method itself*.
        //
        // As an example, if the user wrote something like
        // `foo.bar::<u32>(...)` -- the `Self` type here will be the
        // type of `foo` (possibly adjusted), but we don't want to
        // include that. We want just the `[_, u32]` part.
        if !substs.is_empty() && !generics.params.is_empty() {
            let user_type_annotation = self.probe(|_| {
                let user_substs = UserSubsts {
                    substs: InternalSubsts::for_item(self.tcx, pick.item.def_id, |param, _| {
                        let i = param.index as usize;
                        if i < generics.parent_count {
                            self.fcx.var_for_def(DUMMY_SP, param)
                        } else {
                            substs[i]
                        }
                    }),
                    user_self_ty: None, // not relevant here
                };

                self.fcx.canonicalize_user_type_annotation(UserType::TypeOf(
                    pick.item.def_id,
                    user_substs,
                ))
            });

            debug!("instantiate_method_substs: user_type_annotation={:?}", user_type_annotation);

            if !self.skip_record_for_diagnostics {
                self.fcx.write_user_type_annotation(self.call_expr.hir_id, user_type_annotation);
            }
        }

        self.normalize(self.span, substs)
    }

    fn unify_receivers(
        &mut self,
        self_ty: Ty<'tcx>,
        method_self_ty: Ty<'tcx>,
        pick: &probe::Pick<'tcx>,
        substs: SubstsRef<'tcx>,
    ) {
        debug!(
            "unify_receivers: self_ty={:?} method_self_ty={:?} span={:?} pick={:?}",
            self_ty, method_self_ty, self.span, pick
        );
        let cause = self.cause(
            self.self_expr.span,
            ObligationCauseCode::UnifyReceiver(Box::new(UnifyReceiverContext {
                assoc_item: pick.item,
                param_env: self.param_env,
                substs,
            })),
        );
        match self.at(&cause, self.param_env).sup(DefineOpaqueTypes::No, method_self_ty, self_ty) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
            }
            Err(terr) => {
                // FIXME(arbitrary_self_types): We probably should limit the
                // situations where this can occur by adding additional restrictions
                // to the feature, like the self type can't reference method substs.
                if self.tcx.features().arbitrary_self_types {
                    self.err_ctxt()
                        .report_mismatched_types(&cause, method_self_ty, self_ty, terr)
                        .emit();
                } else {
                    span_bug!(
                        self.span,
                        "{} was a subtype of {} but now is not?",
                        self_ty,
                        method_self_ty
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
        all_substs: SubstsRef<'tcx>,
    ) -> (ty::FnSig<'tcx>, ty::InstantiatedPredicates<'tcx>) {
        debug!("instantiate_method_sig(pick={:?}, all_substs={:?})", pick, all_substs);

        // Instantiate the bounds on the method with the
        // type/early-bound-regions substitutions performed. There can
        // be no late-bound regions appearing here.
        let def_id = pick.item.def_id;
        let method_predicates = self.tcx.predicates_of(def_id).instantiate(self.tcx, all_substs);

        debug!("method_predicates after subst = {:?}", method_predicates);

        let sig = self.tcx.fn_sig(def_id).subst(self.tcx, all_substs);
        debug!("type scheme substituted, sig={:?}", sig);

        let sig = self.instantiate_binder_with_fresh_vars(sig);
        debug!("late-bound lifetimes from method instantiated, sig={:?}", sig);

        (sig, method_predicates)
    }

    fn add_obligations(
        &mut self,
        fty: Ty<'tcx>,
        all_substs: SubstsRef<'tcx>,
        method_predicates: ty::InstantiatedPredicates<'tcx>,
        def_id: DefId,
    ) {
        debug!(
            "add_obligations: fty={:?} all_substs={:?} method_predicates={:?} def_id={:?}",
            fty, all_substs, method_predicates, def_id
        );

        // FIXME: could replace with the following, but we already calculated `method_predicates`,
        // so we just call `predicates_for_generics` directly to avoid redoing work.
        // `self.add_required_obligations(self.span, def_id, &all_substs);`
        for obligation in traits::predicates_for_generics(
            |idx, span| {
                let code = if span.is_dummy() {
                    ObligationCauseCode::ExprItemObligation(def_id, self.call_expr.hir_id, idx)
                } else {
                    ObligationCauseCode::ExprBindingObligation(
                        def_id,
                        span,
                        self.call_expr.hir_id,
                        idx,
                    )
                };
                traits::ObligationCause::new(self.span, self.body_id, code)
            },
            self.param_env,
            method_predicates,
        ) {
            self.register_predicate(obligation);
        }

        // this is a projection from a trait reference, so we have to
        // make sure that the trait reference inputs are well-formed.
        self.add_wf_bounds(all_substs, self.call_expr);

        // the function type must also be well-formed (this is not
        // implied by the substs being well-formed because of inherent
        // impls and late-bound regions - see issue #28609).
        self.register_wf_obligation(fty.into(), self.span, traits::WellFormed(None));
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
                        .unwrap_or(rustc_span::DUMMY_SP);
                    Some((trait_pred, span))
                }
                _ => None,
            })
            .find_map(|(trait_pred, span)| match trait_pred.self_ty().kind() {
                ty::Dynamic(..) => Some(span),
                _ => None,
            })
    }

    fn enforce_illegal_method_limitations(&self, pick: &probe::Pick<'_>) {
        // Disallow calls to the method `drop` defined in the `Drop` trait.
        if let Some(trait_def_id) = pick.item.trait_container(self.tcx) {
            callee::check_legal_trait_for_method_call(
                self.tcx,
                self.span,
                Some(self.self_expr.span),
                self.call_expr.span,
                trait_def_id,
            )
        }
    }

    fn upcast(
        &mut self,
        source_trait_ref: ty::PolyTraitRef<'tcx>,
        target_trait_def_id: DefId,
    ) -> ty::PolyTraitRef<'tcx> {
        let upcast_trait_refs =
            traits::upcast_choices(self.tcx, source_trait_ref, target_trait_def_id);

        // must be exactly one trait ref or we'd get an ambig error etc
        if upcast_trait_refs.len() != 1 {
            span_bug!(
                self.span,
                "cannot uniquely upcast `{:?}` to `{:?}`: `{:?}`",
                source_trait_ref,
                target_trait_def_id,
                upcast_trait_refs
            );
        }

        upcast_trait_refs.into_iter().next().unwrap()
    }

    fn instantiate_binder_with_fresh_vars<T>(&self, value: ty::Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        self.fcx.instantiate_binder_with_fresh_vars(self.span, infer::FnCall, value)
    }
}
