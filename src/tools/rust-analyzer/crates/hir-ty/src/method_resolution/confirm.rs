//! Confirmation step of method selection, meaning ensuring the selected candidate
//! is valid and registering all obligations.

use hir_def::{
    FunctionId, GenericDefId, GenericParamId, ItemContainerId, TraitId,
    expr_store::path::{GenericArg as HirGenericArg, GenericArgs as HirGenericArgs},
    hir::{ExprId, generics::GenericParamDataRef},
};
use rustc_type_ir::{
    TypeFoldable,
    elaborate::elaborate,
    inherent::{BoundExistentialPredicates, IntoKind, SliceLike, Ty as _},
};
use tracing::debug;

use crate::{
    Adjust, Adjustment, AutoBorrow, IncorrectGenericsLenKind, InferenceDiagnostic,
    LifetimeElisionKind, PointerCast,
    db::HirDatabase,
    infer::{AllowTwoPhase, AutoBorrowMutability, InferenceContext, TypeMismatch},
    lower::{
        GenericPredicates,
        path::{GenericArgsLowerer, TypeLikeConst, substs_from_args_and_bindings},
    },
    method_resolution::{CandidateId, MethodCallee, probe},
    next_solver::{
        Binder, Clause, ClauseKind, Const, DbInterner, EarlyParamRegion, ErrorGuaranteed, FnSig,
        GenericArg, GenericArgs, ParamConst, PolyExistentialTraitRef, PolyTraitRef, Region,
        TraitRef, Ty, TyKind,
        infer::{
            BoundRegionConversionTime, InferCtxt,
            traits::{ObligationCause, PredicateObligation},
        },
        util::{clauses_as_obligations, upcast_choices},
    },
};

struct ConfirmContext<'a, 'b, 'db> {
    ctx: &'a mut InferenceContext<'b, 'db>,
    candidate: FunctionId,
    expr: ExprId,
}

#[derive(Debug)]
pub(crate) struct ConfirmResult<'db> {
    pub(crate) callee: MethodCallee<'db>,
    pub(crate) illegal_sized_bound: bool,
    pub(crate) adjustments: Box<[Adjustment<'db>]>,
}

impl<'a, 'db> InferenceContext<'a, 'db> {
    pub(crate) fn confirm_method(
        &mut self,
        pick: &probe::Pick<'db>,
        unadjusted_self_ty: Ty<'db>,
        expr: ExprId,
        generic_args: Option<&HirGenericArgs>,
    ) -> ConfirmResult<'db> {
        debug!(
            "confirm(unadjusted_self_ty={:?}, pick={:?}, generic_args={:?})",
            unadjusted_self_ty, pick, generic_args,
        );

        let CandidateId::FunctionId(candidate) = pick.item else {
            panic!("confirmation is only done for method calls, not path lookups");
        };
        let mut confirm_cx = ConfirmContext::new(self, candidate, expr);
        confirm_cx.confirm(unadjusted_self_ty, pick, generic_args)
    }
}

impl<'a, 'b, 'db> ConfirmContext<'a, 'b, 'db> {
    fn new(
        ctx: &'a mut InferenceContext<'b, 'db>,
        candidate: FunctionId,
        expr: ExprId,
    ) -> ConfirmContext<'a, 'b, 'db> {
        ConfirmContext { ctx, candidate, expr }
    }

    #[inline]
    fn db(&self) -> &'db dyn HirDatabase {
        self.ctx.table.infer_ctxt.interner.db
    }

    #[inline]
    fn interner(&self) -> DbInterner<'db> {
        self.ctx.table.infer_ctxt.interner
    }

    #[inline]
    fn infcx(&self) -> &InferCtxt<'db> {
        &self.ctx.table.infer_ctxt
    }

    fn confirm(
        &mut self,
        unadjusted_self_ty: Ty<'db>,
        pick: &probe::Pick<'db>,
        generic_args: Option<&HirGenericArgs>,
    ) -> ConfirmResult<'db> {
        // Adjust the self expression the user provided and obtain the adjusted type.
        let (self_ty, adjustments) = self.adjust_self_ty(unadjusted_self_ty, pick);

        // Create generic args for the method's type parameters.
        let rcvr_args = self.fresh_receiver_args(self_ty, pick);
        let all_args = self.instantiate_method_args(generic_args, rcvr_args);

        debug!("rcvr_args={rcvr_args:?}, all_args={all_args:?}");

        // Create the final signature for the method, replacing late-bound regions.
        let (method_sig, method_predicates) =
            self.instantiate_method_sig(pick, all_args.as_slice());

        // If there is a `Self: Sized` bound and `Self` is a trait object, it is possible that
        // something which derefs to `Self` actually implements the trait and the caller
        // wanted to make a static dispatch on it but forgot to import the trait.
        // See test `tests/ui/issues/issue-35976.rs`.
        //
        // In that case, we'll error anyway, but we'll also re-run the search with all traits
        // in scope, and if we find another method which can be used, we'll output an
        // appropriate hint suggesting to import the trait.
        let filler_args = GenericArgs::fill_rest(
            self.interner(),
            self.candidate.into(),
            rcvr_args,
            |index, id, _| match id {
                GenericParamId::TypeParamId(id) => Ty::new_param(self.interner(), id, index).into(),
                GenericParamId::ConstParamId(id) => {
                    Const::new_param(self.interner(), ParamConst { id, index }).into()
                }
                GenericParamId::LifetimeParamId(id) => {
                    Region::new_early_param(self.interner(), EarlyParamRegion { id, index }).into()
                }
            },
        );
        let illegal_sized_bound = self.predicates_require_illegal_sized_bound(
            GenericPredicates::query_all(self.db(), self.candidate.into())
                .iter_instantiated_copied(self.interner(), filler_args.as_slice()),
        );

        // Unify the (adjusted) self type with what the method expects.
        //
        // SUBTLE: if we want good error messages, because of "guessing" while matching
        // traits, no trait system method can be called before this point because they
        // could alter our Self-type, except for normalizing the receiver from the
        // signature (which is also done during probing).
        let method_sig_rcvr = method_sig.inputs().as_slice()[0];
        debug!(
            "confirm: self_ty={:?} method_sig_rcvr={:?} method_sig={:?}",
            self_ty, method_sig_rcvr, method_sig
        );
        self.unify_receivers(self_ty, method_sig_rcvr, pick);

        // Make sure nobody calls `drop()` explicitly.
        self.check_for_illegal_method_calls();

        // Lint when an item is shadowing a supertrait item.
        self.lint_shadowed_supertrait_items(pick);

        // Add any trait/regions obligations specified on the method's type parameters.
        // We won't add these if we encountered an illegal sized bound, so that we can use
        // a custom error in that case.
        if !illegal_sized_bound {
            self.add_obligations(method_sig, all_args, method_predicates);
        }

        // Create the final `MethodCallee`.
        let callee = MethodCallee { def_id: self.candidate, args: all_args, sig: method_sig };
        ConfirmResult { callee, illegal_sized_bound, adjustments }
    }

    ///////////////////////////////////////////////////////////////////////////
    // ADJUSTMENTS

    fn adjust_self_ty(
        &mut self,
        unadjusted_self_ty: Ty<'db>,
        pick: &probe::Pick<'db>,
    ) -> (Ty<'db>, Box<[Adjustment<'db>]>) {
        // Commit the autoderefs by calling `autoderef` again, but this
        // time writing the results into the various typeck results.
        let mut autoderef = self.ctx.table.autoderef_with_tracking(unadjusted_self_ty);
        let Some((mut target, n)) = autoderef.nth(pick.autoderefs) else {
            return (Ty::new_error(self.interner(), ErrorGuaranteed), Box::new([]));
        };
        assert_eq!(n, pick.autoderefs);

        let mut adjustments =
            self.ctx.table.register_infer_ok(autoderef.adjust_steps_as_infer_ok());
        match pick.autoref_or_ptr_adjustment {
            Some(probe::AutorefOrPtrAdjustment::Autoref { mutbl, unsize }) => {
                let region = self.infcx().next_region_var();
                // Type we're wrapping in a reference, used later for unsizing
                let base_ty = target;

                target = Ty::new_ref(self.interner(), region, target, mutbl);

                // Method call receivers are the primary use case
                // for two-phase borrows.
                let mutbl = AutoBorrowMutability::new(mutbl, AllowTwoPhase::Yes);

                adjustments
                    .push(Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)), target });

                if unsize {
                    let unsized_ty = if let TyKind::Array(elem_ty, _) = base_ty.kind() {
                        Ty::new_slice(self.interner(), elem_ty)
                    } else {
                        panic!(
                            "AutorefOrPtrAdjustment's unsize flag should only be set for array ty, found {:?}",
                            base_ty
                        )
                    };
                    target = Ty::new_ref(self.interner(), region, unsized_ty, mutbl.into());
                    adjustments
                        .push(Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), target });
                }
            }
            Some(probe::AutorefOrPtrAdjustment::ToConstPtr) => {
                target = match target.kind() {
                    TyKind::RawPtr(ty, mutbl) => {
                        assert!(mutbl.is_mut());
                        Ty::new_imm_ptr(self.interner(), ty)
                    }
                    other => panic!("Cannot adjust receiver type {other:?} to const ptr"),
                };

                adjustments.push(Adjustment {
                    kind: Adjust::Pointer(PointerCast::MutToConstPointer),
                    target,
                });
            }
            None => {}
        }

        (target, adjustments.into_boxed_slice())
    }

    /// Returns a set of generic parameters for the method *receiver* where all type and region
    /// parameters are instantiated with fresh variables. This generic parameters does not include any
    /// parameters declared on the method itself.
    ///
    /// Note that this generic parameters may include late-bound regions from the impl level. If so,
    /// these are instantiated later in the `instantiate_method_sig` routine.
    fn fresh_receiver_args(
        &mut self,
        self_ty: Ty<'db>,
        pick: &probe::Pick<'db>,
    ) -> GenericArgs<'db> {
        match pick.kind {
            probe::InherentImplPick(impl_def_id) => {
                self.infcx().fresh_args_for_item(impl_def_id.into())
            }

            probe::ObjectPick(trait_def_id) => {
                // If the trait is not object safe (specifically, we care about when
                // the receiver is not valid), then there's a chance that we will not
                // actually be able to recover the object by derefing the receiver like
                // we should if it were valid.
                if self.db().dyn_compatibility_of_trait(trait_def_id).is_some() {
                    return GenericArgs::error_for_item(self.interner(), trait_def_id.into());
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
                    let original_poly_trait_ref =
                        principal.with_self_ty(this.interner(), object_ty);
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

            probe::TraitPick(trait_def_id) => {
                // Make a trait reference `$0 : Trait<$1...$n>`
                // consisting entirely of type variables. Later on in
                // the process we will unify the transformed-self-type
                // of the method with the actual type in order to
                // unify some of these variables.
                self.infcx().fresh_args_for_item(trait_def_id.into())
            }

            probe::WhereClausePick(poly_trait_ref) => {
                // Where clauses can have bound regions in them. We need to instantiate
                // those to convert from a poly-trait-ref to a trait-ref.
                self.instantiate_binder_with_fresh_vars(poly_trait_ref).args
            }
        }
    }

    fn extract_existential_trait_ref<R, F>(&self, self_ty: Ty<'db>, mut closure: F) -> R
    where
        F: FnMut(&ConfirmContext<'a, 'b, 'db>, Ty<'db>, PolyExistentialTraitRef<'db>) -> R,
    {
        // If we specified that this is an object method, then the
        // self-type ought to be something that can be dereferenced to
        // yield an object-type (e.g., `&Object` or `Box<Object>`
        // etc).

        let mut autoderef = self.ctx.table.autoderef(self_ty);

        // We don't need to gate this behind arbitrary self types
        // per se, but it does make things a bit more gated.
        if self.ctx.unstable_features.arbitrary_self_types
            || self.ctx.unstable_features.arbitrary_self_types_pointers
        {
            autoderef = autoderef.use_receiver_trait();
        }

        autoderef
            .include_raw_pointers()
            .find_map(|(ty, _)| match ty.kind() {
                TyKind::Dynamic(data, ..) => Some(closure(
                    self,
                    ty,
                    data.principal().expect("calling trait method on empty object?"),
                )),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!("self-type `{:?}` for ObjectPick never dereferenced to an object", self_ty)
            })
    }

    fn instantiate_method_args(
        &mut self,
        generic_args: Option<&HirGenericArgs>,
        parent_args: GenericArgs<'db>,
    ) -> GenericArgs<'db> {
        struct LowererCtx<'a, 'b, 'db> {
            ctx: &'a mut InferenceContext<'b, 'db>,
            expr: ExprId,
            parent_args: &'a [GenericArg<'db>],
        }

        impl<'db> GenericArgsLowerer<'db> for LowererCtx<'_, '_, 'db> {
            fn report_len_mismatch(
                &mut self,
                def: GenericDefId,
                provided_count: u32,
                expected_count: u32,
                kind: IncorrectGenericsLenKind,
            ) {
                self.ctx.push_diagnostic(InferenceDiagnostic::MethodCallIncorrectGenericsLen {
                    expr: self.expr,
                    provided_count,
                    expected_count,
                    kind,
                    def,
                });
            }

            fn report_arg_mismatch(
                &mut self,
                param_id: GenericParamId,
                arg_idx: u32,
                has_self_arg: bool,
            ) {
                self.ctx.push_diagnostic(InferenceDiagnostic::MethodCallIncorrectGenericsOrder {
                    expr: self.expr,
                    param_id,
                    arg_idx,
                    has_self_arg,
                });
            }

            fn provided_kind(
                &mut self,
                param_id: GenericParamId,
                param: GenericParamDataRef<'_>,
                arg: &HirGenericArg,
            ) -> GenericArg<'db> {
                match (param, arg) {
                    (
                        GenericParamDataRef::LifetimeParamData(_),
                        HirGenericArg::Lifetime(lifetime),
                    ) => self.ctx.make_body_lifetime(*lifetime).into(),
                    (GenericParamDataRef::TypeParamData(_), HirGenericArg::Type(type_ref)) => {
                        self.ctx.make_body_ty(*type_ref).into()
                    }
                    (GenericParamDataRef::ConstParamData(_), HirGenericArg::Const(konst)) => {
                        let GenericParamId::ConstParamId(const_id) = param_id else {
                            unreachable!("non-const param ID for const param");
                        };
                        let const_ty = self.ctx.db.const_param_ty_ns(const_id);
                        self.ctx.make_body_const(*konst, const_ty).into()
                    }
                    _ => unreachable!("unmatching param kinds were passed to `provided_kind()`"),
                }
            }

            fn provided_type_like_const(
                &mut self,
                const_ty: Ty<'db>,
                arg: TypeLikeConst<'_>,
            ) -> Const<'db> {
                match arg {
                    TypeLikeConst::Path(path) => self.ctx.make_path_as_body_const(path, const_ty),
                    TypeLikeConst::Infer => self.ctx.table.next_const_var(),
                }
            }

            fn inferred_kind(
                &mut self,
                _def: GenericDefId,
                param_id: GenericParamId,
                _param: GenericParamDataRef<'_>,
                _infer_args: bool,
                _preceding_args: &[GenericArg<'db>],
            ) -> GenericArg<'db> {
                // Always create an inference var, even when `infer_args == false`. This helps with diagnostics,
                // and I think it's also required in the presence of `impl Trait` (that must be inferred).
                self.ctx.table.next_var_for_param(param_id)
            }

            fn parent_arg(&mut self, param_idx: u32, _param_id: GenericParamId) -> GenericArg<'db> {
                self.parent_args[param_idx as usize]
            }

            fn report_elided_lifetimes_in_path(
                &mut self,
                _def: GenericDefId,
                _expected_count: u32,
                _hard_error: bool,
            ) {
                unreachable!("we set `LifetimeElisionKind::Infer`")
            }

            fn report_elision_failure(&mut self, _def: GenericDefId, _expected_count: u32) {
                unreachable!("we set `LifetimeElisionKind::Infer`")
            }

            fn report_missing_lifetime(&mut self, _def: GenericDefId, _expected_count: u32) {
                unreachable!("we set `LifetimeElisionKind::Infer`")
            }
        }

        substs_from_args_and_bindings(
            self.db(),
            self.ctx.body,
            generic_args,
            self.candidate.into(),
            true,
            LifetimeElisionKind::Infer,
            false,
            None,
            &mut LowererCtx { ctx: self.ctx, expr: self.expr, parent_args: parent_args.as_slice() },
        )
    }

    fn unify_receivers(
        &mut self,
        self_ty: Ty<'db>,
        method_self_ty: Ty<'db>,
        pick: &probe::Pick<'db>,
    ) {
        debug!(
            "unify_receivers: self_ty={:?} method_self_ty={:?} pick={:?}",
            self_ty, method_self_ty, pick
        );
        let cause = ObligationCause::new();
        match self.ctx.table.at(&cause).sup(method_self_ty, self_ty) {
            Ok(infer_ok) => {
                self.ctx.table.register_infer_ok(infer_ok);
            }
            Err(_) => {
                if self.ctx.unstable_features.arbitrary_self_types {
                    self.ctx.result.type_mismatches.get_or_insert_default().insert(
                        self.expr.into(),
                        TypeMismatch { expected: method_self_ty, actual: self_ty },
                    );
                }
            }
        }
    }

    // NOTE: this returns the *unnormalized* predicates and method sig. Because of
    // inference guessing, the predicates and method signature can't be normalized
    // until we unify the `Self` type.
    fn instantiate_method_sig<'c>(
        &mut self,
        pick: &probe::Pick<'db>,
        all_args: &'c [GenericArg<'db>],
    ) -> (FnSig<'db>, impl Iterator<Item = PredicateObligation<'db>> + use<'c, 'db>) {
        debug!("instantiate_method_sig(pick={:?}, all_args={:?})", pick, all_args);

        // Instantiate the bounds on the method with the
        // type/early-bound-regions instantiations performed. There can
        // be no late-bound regions appearing here.
        let def_id = self.candidate;
        let method_predicates = clauses_as_obligations(
            GenericPredicates::query_all(self.db(), def_id.into())
                .iter_instantiated_copied(self.interner(), all_args),
            ObligationCause::new(),
            self.ctx.table.param_env,
        );

        let sig =
            self.db().callable_item_signature(def_id.into()).instantiate(self.interner(), all_args);
        debug!("type scheme instantiated, sig={:?}", sig);

        let sig = self.instantiate_binder_with_fresh_vars(sig);
        debug!("late-bound lifetimes from method instantiated, sig={:?}", sig);

        (sig, method_predicates)
    }

    fn add_obligations(
        &mut self,
        sig: FnSig<'db>,
        all_args: GenericArgs<'db>,
        method_predicates: impl Iterator<Item = PredicateObligation<'db>>,
    ) {
        debug!("add_obligations: sig={:?} all_args={:?}", sig, all_args);

        self.ctx.table.register_predicates(method_predicates);

        // this is a projection from a trait reference, so we have to
        // make sure that the trait reference inputs are well-formed.
        self.ctx.table.add_wf_bounds(all_args);

        // the function type must also be well-formed (this is not
        // implied by the args being well-formed because of inherent
        // impls and late-bound regions - see issue #28609).
        for ty in sig.inputs_and_output {
            self.ctx.table.register_wf_obligation(ty.into(), ObligationCause::new());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY

    fn predicates_require_illegal_sized_bound(
        &self,
        predicates: impl Iterator<Item = Clause<'db>>,
    ) -> bool {
        let Some(sized_def_id) = self.ctx.lang_items.Sized else {
            return false;
        };

        elaborate(self.interner(), predicates)
            // We don't care about regions here.
            .filter_map(|pred| match pred.kind().skip_binder() {
                ClauseKind::Trait(trait_pred) if trait_pred.def_id().0 == sized_def_id => {
                    Some(trait_pred)
                }
                _ => None,
            })
            .any(|trait_pred| matches!(trait_pred.self_ty().kind(), TyKind::Dynamic(..)))
    }

    fn check_for_illegal_method_calls(&self) {
        // Disallow calls to the method `drop` defined in the `Drop` trait.
        if let ItemContainerId::TraitId(trait_def_id) = self.candidate.loc(self.db()).container
            && self.ctx.lang_items.Drop.is_some_and(|drop_trait| drop_trait == trait_def_id)
        {
            // FIXME: Report an error.
        }
    }

    #[expect(clippy::needless_return)]
    fn lint_shadowed_supertrait_items(&self, pick: &probe::Pick<'_>) {
        if pick.shadowed_candidates.is_empty() {
            return;
        }

        // FIXME: Emit the lint.
    }

    fn upcast(
        &self,
        source_trait_ref: PolyTraitRef<'db>,
        target_trait_def_id: TraitId,
    ) -> PolyTraitRef<'db> {
        let upcast_trait_refs =
            upcast_choices(self.interner(), source_trait_ref, target_trait_def_id);

        // must be exactly one trait ref or we'd get an ambig error etc
        if let &[upcast_trait_ref] = upcast_trait_refs.as_slice() {
            upcast_trait_ref
        } else {
            Binder::dummy(TraitRef::new_from_args(
                self.interner(),
                target_trait_def_id.into(),
                GenericArgs::error_for_item(self.interner(), target_trait_def_id.into()),
            ))
        }
    }

    fn instantiate_binder_with_fresh_vars<T>(&self, value: Binder<'db, T>) -> T
    where
        T: TypeFoldable<DbInterner<'db>> + Copy,
    {
        self.infcx().instantiate_binder_with_fresh_vars(BoundRegionConversionTime::FnCall, value)
    }
}
