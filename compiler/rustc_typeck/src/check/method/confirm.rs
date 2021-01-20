use super::{probe, MethodCallee};

use crate::astconv::{AstConv, CreateSubstsForGenericArgsCtxt};
use crate::check::{callee, FnCtxt};
use crate::hir::def_id::DefId;
use crate::hir::GenericArg;
use rustc_hir as hir;
use rustc_infer::infer::{self, InferOk};
use rustc_middle::traits::{ObligationCauseCode, UnifyReceiverContext};
use rustc_middle::ty::adjustment::{Adjust, Adjustment, PointerCast};
use rustc_middle::ty::adjustment::{AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::{self, Subst, SubstsRef};
use rustc_middle::ty::{self, GenericParamDefKind, Ty};
use rustc_span::Span;
use rustc_trait_selection::traits;

use std::ops::Deref;

struct ConfirmContext<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    self_expr: &'tcx hir::Expr<'tcx>,
    call_expr: &'tcx hir::Expr<'tcx>,
}

impl<'a, 'tcx> Deref for ConfirmContext<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

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
        pick: probe::Pick<'tcx>,
        segment: &hir::PathSegment<'_>,
    ) -> ConfirmResult<'tcx> {
        debug!(
            "confirm(unadjusted_self_ty={:?}, pick={:?}, generic_args={:?})",
            unadjusted_self_ty, pick, segment.args,
        );

        let mut confirm_cx = ConfirmContext::new(self, span, self_expr, call_expr);
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
        ConfirmContext { fcx, span, self_expr, call_expr }
    }

    fn confirm(
        &mut self,
        unadjusted_self_ty: Ty<'tcx>,
        pick: probe::Pick<'tcx>,
        segment: &hir::PathSegment<'_>,
    ) -> ConfirmResult<'tcx> {
        // Adjust the self expression the user provided and obtain the adjusted type.
        let self_ty = self.adjust_self_ty(unadjusted_self_ty, &pick);

        // Create substitutions for the method's type parameters.
        let rcvr_substs = self.fresh_receiver_substs(self_ty, &pick);
        let all_substs = self.instantiate_method_substs(&pick, segment, rcvr_substs);

        debug!("all_substs={:?}", all_substs);

        // Create the final signature for the method, replacing late-bound regions.
        let (method_sig, method_predicates) = self.instantiate_method_sig(&pick, all_substs);

        // Unify the (adjusted) self type with what the method expects.
        //
        // SUBTLE: if we want good error messages, because of "guessing" while matching
        // traits, no trait system method can be called before this point because they
        // could alter our Self-type, except for normalizing the receiver from the
        // signature (which is also done during probing).
        let method_sig_rcvr = self.normalize_associated_types_in(self.span, method_sig.inputs()[0]);
        debug!(
            "confirm: self_ty={:?} method_sig_rcvr={:?} method_sig={:?} method_predicates={:?}",
            self_ty, method_sig_rcvr, method_sig, method_predicates
        );
        self.unify_receivers(self_ty, method_sig_rcvr, &pick, all_substs);

        let (method_sig, method_predicates) =
            self.normalize_associated_types_in(self.span, (method_sig, method_predicates));

        // Make sure nobody calls `drop()` explicitly.
        self.enforce_illegal_method_limitations(&pick);

        // If there is a `Self: Sized` bound and `Self` is a trait object, it is possible that
        // something which derefs to `Self` actually implements the trait and the caller
        // wanted to make a static dispatch on it but forgot to import the trait.
        // See test `src/test/ui/issue-35976.rs`.
        //
        // In that case, we'll error anyway, but we'll also re-run the search with all traits
        // in scope, and if we find another method which can be used, we'll output an
        // appropriate hint suggesting to import the trait.
        let illegal_sized_bound = self.predicates_require_illegal_sized_bound(&method_predicates);

        // Add any trait/regions obligations specified on the method's type parameters.
        // We won't add these if we encountered an illegal sized bound, so that we can use
        // a custom error in that case.
        if illegal_sized_bound.is_none() {
            let method_ty = self.tcx.mk_fn_ptr(ty::Binder::bind(method_sig));
            self.add_obligations(method_ty, all_substs, method_predicates);
        }

        // Create the final `MethodCallee`.
        let callee = MethodCallee { def_id: pick.item.def_id, substs: all_substs, sig: method_sig };
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
        let mut autoderef =
            self.autoderef_overloaded_span(self.span, unadjusted_self_ty, self.call_expr.span);
        let (_, n) = match autoderef.nth(pick.autoderefs) {
            Some(n) => n,
            None => {
                return self.tcx.ty_error_with_message(
                    rustc_span::DUMMY_SP,
                    &format!("failed autoderef {}", pick.autoderefs),
                );
            }
        };
        assert_eq!(n, pick.autoderefs);

        let mut adjustments = self.adjust_steps(&autoderef);

        let mut target =
            self.structurally_resolved_type(autoderef.span(), autoderef.final_ty(false));

        if let Some(mutbl) = pick.autoref {
            let region = self.next_region_var(infer::Autoref(self.span, pick.item));
            target = self.tcx.mk_ref(region, ty::TypeAndMut { mutbl, ty: target });
            let mutbl = match mutbl {
                hir::Mutability::Not => AutoBorrowMutability::Not,
                hir::Mutability::Mut => AutoBorrowMutability::Mut {
                    // Method call receivers are the primary use case
                    // for two-phase borrows.
                    allow_two_phase_borrow: AllowTwoPhase::Yes,
                },
            };
            adjustments
                .push(Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)), target });

            if let Some(unsize_target) = pick.unsize {
                target = self
                    .tcx
                    .mk_ref(region, ty::TypeAndMut { mutbl: mutbl.into(), ty: unsize_target });
                adjustments.push(Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), target });
            }
        } else {
            // No unsizing should be performed without autoref (at
            // least during method dispach). This is because we
            // currently only unsize `[T;N]` to `[T]`, and naturally
            // that must occur being a reference.
            assert!(pick.unsize.is_none());
        }

        self.register_predicates(autoderef.into_obligations());

        // Write out the final adjustments.
        self.apply_adjustments(self.self_expr, adjustments);

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
                let impl_def_id = pick.item.container.id();
                assert!(
                    self.tcx.impl_trait_ref(impl_def_id).is_none(),
                    "impl {:?} is not an inherent impl",
                    impl_def_id
                );
                self.fresh_substs_for_item(self.span, impl_def_id)
            }

            probe::ObjectPick => {
                let trait_def_id = pick.item.container.id();
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
                        this.replace_bound_vars_with_fresh_vars(upcast_poly_trait_ref);
                    debug!(
                        "original_poly_trait_ref={:?} upcast_trait_ref={:?} target_trait={:?}",
                        original_poly_trait_ref, upcast_trait_ref, trait_def_id
                    );
                    upcast_trait_ref.substs
                })
            }

            probe::TraitPick => {
                let trait_def_id = pick.item.container.id();

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
                self.replace_bound_vars_with_fresh_vars(poly_trait_ref).substs
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
                ty::Dynamic(ref data, ..) => Some(closure(
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
        let arg_count_correct = AstConv::check_generic_arg_count_for_call(
            self.tcx, self.span, &generics, &seg, true, // `is_method_call`
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
                    if let Some(ref data) = self.seg.args {
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
                        AstConv::ast_region_to_region(self.cfcx.fcx, lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.cfcx.to_ty(ty).into()
                    }
                    (GenericParamDefKind::Const, GenericArg::Const(ct)) => {
                        self.cfcx.const_arg_to_const(&ct.value, param.def_id).into()
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
        AstConv::create_substs_for_generic_args(
            self.tcx,
            pick.item.def_id,
            parent_substs,
            false,
            None,
            arg_count_correct,
            &mut MethodSubstsCtxt { cfcx: self, pick, seg },
        )
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
            self.span,
            ObligationCauseCode::UnifyReceiver(Box::new(UnifyReceiverContext {
                assoc_item: pick.item,
                param_env: self.param_env,
                substs,
            })),
        );
        match self.at(&cause, self.param_env).sup(method_self_ty, self_ty) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
            }
            Err(_) => {
                span_bug!(
                    self.span,
                    "{} was a subtype of {} but now is not?",
                    self_ty,
                    method_self_ty
                );
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

        let sig = self.tcx.fn_sig(def_id);

        // Instantiate late-bound regions and substitute the trait
        // parameters into the method type to get the actual method type.
        //
        // N.B., instantiate late-bound regions first so that
        // `instantiate_type_scheme` can normalize associated types that
        // may reference those regions.
        let method_sig = self.replace_bound_vars_with_fresh_vars(sig);
        debug!("late-bound lifetimes from method instantiated, method_sig={:?}", method_sig);

        let method_sig = method_sig.subst(self.tcx, all_substs);
        debug!("type scheme substituted, method_sig={:?}", method_sig);

        (method_sig, method_predicates)
    }

    fn add_obligations(
        &mut self,
        fty: Ty<'tcx>,
        all_substs: SubstsRef<'tcx>,
        method_predicates: ty::InstantiatedPredicates<'tcx>,
    ) {
        debug!(
            "add_obligations: fty={:?} all_substs={:?} method_predicates={:?}",
            fty, all_substs, method_predicates
        );

        self.add_obligations_for_parameters(
            traits::ObligationCause::misc(self.span, self.body_id),
            method_predicates,
        );

        // this is a projection from a trait reference, so we have to
        // make sure that the trait reference inputs are well-formed.
        self.add_wf_bounds(all_substs, self.call_expr);

        // the function type must also be well-formed (this is not
        // implied by the substs being well-formed because of inherent
        // impls and late-bound regions - see issue #28609).
        self.register_wf_obligation(fty.into(), self.span, traits::MiscObligation);
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY

    fn predicates_require_illegal_sized_bound(
        &self,
        predicates: &ty::InstantiatedPredicates<'tcx>,
    ) -> Option<Span> {
        let sized_def_id = match self.tcx.lang_items().sized_trait() {
            Some(def_id) => def_id,
            None => return None,
        };

        traits::elaborate_predicates(self.tcx, predicates.predicates.iter().copied())
            // We don't care about regions here.
            .filter_map(|obligation| match obligation.predicate.skip_binders() {
                ty::PredicateAtom::Trait(trait_pred, _) if trait_pred.def_id() == sized_def_id => {
                    let span = predicates
                        .predicates
                        .iter()
                        .zip(predicates.spans.iter())
                        .find_map(
                            |(p, span)| {
                                if *p == obligation.predicate { Some(*span) } else { None }
                            },
                        )
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
        match pick.item.container {
            ty::TraitContainer(trait_def_id) => callee::check_legal_trait_for_method_call(
                self.tcx,
                self.span,
                Some(self.self_expr.span),
                trait_def_id,
            ),
            ty::ImplContainer(..) => {}
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

    fn replace_bound_vars_with_fresh_vars<T>(&self, value: ty::Binder<T>) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.fcx.replace_bound_vars_with_fresh_vars(self.span, infer::FnCall, value).0
    }
}
