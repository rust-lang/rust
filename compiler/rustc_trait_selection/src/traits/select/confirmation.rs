//! Confirmation.
//!
//! Confirmation unifies the output type parameters of the trait
//! with the values found in the obligation, possibly yielding a
//! type error. See the [rustc dev guide] for more details.
//!
//! [rustc dev guide]:
//! https://rustc-dev-guide.rust-lang.org/traits/resolution.html#confirmation

use std::ops::ControlFlow;

use rustc_ast::Mutability;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::{DefineOpaqueTypes, HigherRankedType, InferOk};
use rustc_infer::traits::ObligationCauseCode;
use rustc_middle::traits::{BuiltinImplSource, SignatureMismatchData};
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt, Upcast};
use rustc_middle::{bug, span_bug};
use rustc_span::def_id::DefId;
use rustc_type_ir::elaborate;
use thin_vec::thin_vec;
use tracing::{debug, instrument};

use super::SelectionCandidate::{self, *};
use super::{BuiltinImplConditions, PredicateObligations, SelectionContext};
use crate::traits::normalize::{normalize_with_depth, normalize_with_depth_to};
use crate::traits::util::{self, closure_trait_ref_and_return_type};
use crate::traits::{
    ImplSource, ImplSourceUserDefinedData, Normalized, Obligation, ObligationCause,
    PolyTraitObligation, PredicateObligation, Selection, SelectionError, SignatureMismatch,
    TraitDynIncompatible, TraitObligation, Unimplemented,
};

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    pub(super) fn confirm_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        candidate: SelectionCandidate<'tcx>,
    ) -> Result<Selection<'tcx>, SelectionError<'tcx>> {
        Ok(match candidate {
            SizedCandidate { has_nested } => {
                let data = self.confirm_builtin_candidate(obligation, has_nested);
                ImplSource::Builtin(BuiltinImplSource::Misc, data)
            }

            BuiltinCandidate { has_nested } => {
                let data = self.confirm_builtin_candidate(obligation, has_nested);
                ImplSource::Builtin(BuiltinImplSource::Misc, data)
            }

            TransmutabilityCandidate => {
                let data = self.confirm_transmutability_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, data)
            }

            ParamCandidate(param) => {
                let obligations =
                    self.confirm_param_candidate(obligation, param.map_bound(|t| t.trait_ref));
                ImplSource::Param(obligations)
            }

            ImplCandidate(impl_def_id) => {
                ImplSource::UserDefined(self.confirm_impl_candidate(obligation, impl_def_id))
            }

            AutoImplCandidate => {
                let data = self.confirm_auto_impl_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, data)
            }

            ProjectionCandidate(idx) => {
                let obligations = self.confirm_projection_candidate(obligation, idx)?;
                ImplSource::Param(obligations)
            }

            ObjectCandidate(idx) => self.confirm_object_candidate(obligation, idx)?,

            ClosureCandidate { .. } => {
                let vtable_closure = self.confirm_closure_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, vtable_closure)
            }

            AsyncClosureCandidate => {
                let vtable_closure = self.confirm_async_closure_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, vtable_closure)
            }

            // No nested obligations or confirmation process. The checks that we do in
            // candidate assembly are sufficient.
            AsyncFnKindHelperCandidate => {
                ImplSource::Builtin(BuiltinImplSource::Misc, PredicateObligations::new())
            }

            CoroutineCandidate => {
                let vtable_coroutine = self.confirm_coroutine_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, vtable_coroutine)
            }

            FutureCandidate => {
                let vtable_future = self.confirm_future_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, vtable_future)
            }

            IteratorCandidate => {
                let vtable_iterator = self.confirm_iterator_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, vtable_iterator)
            }

            AsyncIteratorCandidate => {
                let vtable_iterator = self.confirm_async_iterator_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, vtable_iterator)
            }

            FnPointerCandidate => {
                let data = self.confirm_fn_pointer_candidate(obligation)?;
                ImplSource::Builtin(BuiltinImplSource::Misc, data)
            }

            TraitAliasCandidate => {
                let data = self.confirm_trait_alias_candidate(obligation);
                ImplSource::Builtin(BuiltinImplSource::Misc, data)
            }

            BuiltinObjectCandidate => {
                // This indicates something like `Trait + Send: Send`. In this case, we know that
                // this holds because that's what the object type is telling us, and there's really
                // no additional obligations to prove and no types in particular to unify, etc.
                ImplSource::Builtin(BuiltinImplSource::Misc, PredicateObligations::new())
            }

            BuiltinUnsizeCandidate => self.confirm_builtin_unsize_candidate(obligation)?,

            TraitUpcastingUnsizeCandidate(idx) => {
                self.confirm_trait_upcasting_unsize_candidate(obligation, idx)?
            }

            BikeshedGuaranteedNoDropCandidate => {
                self.confirm_bikeshed_guaranteed_no_drop_candidate(obligation)
            }
        })
    }

    fn confirm_projection_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        idx: usize,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let tcx = self.tcx();

        let placeholder_trait_predicate =
            self.infcx.enter_forall_and_leak_universe(obligation.predicate).trait_ref;
        let placeholder_self_ty = self.infcx.shallow_resolve(placeholder_trait_predicate.self_ty());
        let candidate_predicate = self
            .for_each_item_bound(
                placeholder_self_ty,
                |_, clause, clause_idx| {
                    if clause_idx == idx {
                        ControlFlow::Break(clause)
                    } else {
                        ControlFlow::Continue(())
                    }
                },
                || unreachable!(),
            )
            .break_value()
            .expect("expected to index into clause that exists");
        let candidate = candidate_predicate
            .as_trait_clause()
            .expect("projection candidate is not a trait predicate")
            .map_bound(|t| t.trait_ref);

        let candidate = self.infcx.instantiate_binder_with_fresh_vars(
            obligation.cause.span,
            HigherRankedType,
            candidate,
        );
        let mut obligations = PredicateObligations::new();
        let candidate = normalize_with_depth_to(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            candidate,
            &mut obligations,
        );

        obligations.extend(
            self.infcx
                .at(&obligation.cause, obligation.param_env)
                .eq(DefineOpaqueTypes::No, placeholder_trait_predicate, candidate)
                .map(|InferOk { obligations, .. }| obligations)
                .map_err(|_| Unimplemented)?,
        );

        // FIXME(compiler-errors): I don't think this is needed.
        if let ty::Alias(ty::Projection, alias_ty) = placeholder_self_ty.kind() {
            let predicates = tcx.predicates_of(alias_ty.def_id).instantiate_own(tcx, alias_ty.args);
            for (predicate, _) in predicates {
                let normalized = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    predicate,
                    &mut obligations,
                );
                obligations.push(Obligation::with_depth(
                    self.tcx(),
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    normalized,
                ));
            }
        }

        Ok(obligations)
    }

    fn confirm_param_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        param: ty::PolyTraitRef<'tcx>,
    ) -> PredicateObligations<'tcx> {
        debug!(?obligation, ?param, "confirm_param_candidate");

        // During evaluation, we already checked that this
        // where-clause trait-ref could be unified with the obligation
        // trait-ref. Repeat that unification now without any
        // transactional boundary; it should not fail.
        match self.match_where_clause_trait_ref(obligation, param) {
            Ok(obligations) => obligations,
            Err(()) => {
                bug!(
                    "Where clause `{:?}` was applicable to `{:?}` but now is not",
                    param,
                    obligation
                );
            }
        }
    }

    fn confirm_builtin_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        has_nested: bool,
    ) -> PredicateObligations<'tcx> {
        debug!(?obligation, ?has_nested, "confirm_builtin_candidate");

        let tcx = self.tcx();
        let obligations = if has_nested {
            let trait_def = obligation.predicate.def_id();
            let conditions = if tcx.is_lang_item(trait_def, LangItem::Sized) {
                self.sized_conditions(obligation)
            } else if tcx.is_lang_item(trait_def, LangItem::Copy) {
                self.copy_clone_conditions(obligation)
            } else if tcx.is_lang_item(trait_def, LangItem::Clone) {
                self.copy_clone_conditions(obligation)
            } else if tcx.is_lang_item(trait_def, LangItem::FusedIterator) {
                self.fused_iterator_conditions(obligation)
            } else {
                bug!("unexpected builtin trait {:?}", trait_def)
            };
            let BuiltinImplConditions::Where(types) = conditions else {
                bug!("obligation {:?} had matched a builtin impl but now doesn't", obligation);
            };
            let types = self.infcx.enter_forall_and_leak_universe(types);

            let cause = obligation.derived_cause(ObligationCauseCode::BuiltinDerived);
            self.collect_predicates_for_types(
                obligation.param_env,
                cause,
                obligation.recursion_depth + 1,
                trait_def,
                types,
            )
        } else {
            PredicateObligations::new()
        };

        debug!(?obligations);

        obligations
    }

    #[instrument(level = "debug", skip(self))]
    fn confirm_transmutability_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        use rustc_transmute::{Answer, Assume, Condition};

        /// Generate sub-obligations for reference-to-reference transmutations.
        fn reference_obligations<'tcx>(
            tcx: TyCtxt<'tcx>,
            obligation: &PolyTraitObligation<'tcx>,
            (src_lifetime, src_ty, src_mut): (ty::Region<'tcx>, Ty<'tcx>, Mutability),
            (dst_lifetime, dst_ty, dst_mut): (ty::Region<'tcx>, Ty<'tcx>, Mutability),
            assume: Assume,
        ) -> PredicateObligations<'tcx> {
            let make_transmute_obl = |src, dst| {
                let transmute_trait = obligation.predicate.def_id();
                let assume = obligation.predicate.skip_binder().trait_ref.args.const_at(2);
                let trait_ref = ty::TraitRef::new(
                    tcx,
                    transmute_trait,
                    [
                        ty::GenericArg::from(dst),
                        ty::GenericArg::from(src),
                        ty::GenericArg::from(assume),
                    ],
                );
                Obligation::with_depth(
                    tcx,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    obligation.predicate.rebind(trait_ref),
                )
            };

            let make_freeze_obl = |ty| {
                let trait_ref = ty::TraitRef::new(
                    tcx,
                    tcx.require_lang_item(LangItem::Freeze, None),
                    [ty::GenericArg::from(ty)],
                );
                Obligation::with_depth(
                    tcx,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    trait_ref,
                )
            };

            let make_outlives_obl = |target, region| {
                let outlives = ty::OutlivesPredicate(target, region);
                Obligation::with_depth(
                    tcx,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    obligation.predicate.rebind(outlives),
                )
            };

            // Given a transmutation from `&'a (mut) Src` and `&'dst (mut) Dst`,
            // it is always the case that `Src` must be transmutable into `Dst`,
            // and that that `'src` must outlive `'dst`.
            let mut obls = PredicateObligations::with_capacity(1);
            obls.push(make_transmute_obl(src_ty, dst_ty));
            if !assume.lifetimes {
                obls.push(make_outlives_obl(src_lifetime, dst_lifetime));
            }

            // Given a transmutation from `&Src`, both `Src` and `Dst` must be
            // `Freeze`, otherwise, using the transmuted value could lead to
            // data races.
            if src_mut == Mutability::Not {
                obls.extend([make_freeze_obl(src_ty), make_freeze_obl(dst_ty)])
            }

            // Given a transmutation into `&'dst mut Dst`, it also must be the
            // case that `Dst` is transmutable into `Src`. For example,
            // transmuting bool -> u8 is OK as long as you can't update that u8
            // to be > 1, because you could later transmute the u8 back to a
            // bool and get undefined behavior. It also must be the case that
            // `'dst` lives exactly as long as `'src`.
            if dst_mut == Mutability::Mut {
                obls.push(make_transmute_obl(dst_ty, src_ty));
                if !assume.lifetimes {
                    obls.push(make_outlives_obl(dst_lifetime, src_lifetime));
                }
            }

            obls
        }

        /// Flatten the `Condition` tree into a conjunction of obligations.
        #[instrument(level = "debug", skip(tcx, obligation))]
        fn flatten_answer_tree<'tcx>(
            tcx: TyCtxt<'tcx>,
            obligation: &PolyTraitObligation<'tcx>,
            cond: Condition<rustc_transmute::layout::rustc::Ref<'tcx>>,
            assume: Assume,
        ) -> PredicateObligations<'tcx> {
            match cond {
                // FIXME(bryangarza): Add separate `IfAny` case, instead of treating as `IfAll`
                // Not possible until the trait solver supports disjunctions of obligations
                Condition::IfAll(conds) | Condition::IfAny(conds) => conds
                    .into_iter()
                    .flat_map(|cond| flatten_answer_tree(tcx, obligation, cond, assume))
                    .collect(),
                Condition::IfTransmutable { src, dst } => reference_obligations(
                    tcx,
                    obligation,
                    (src.lifetime, src.ty, src.mutability),
                    (dst.lifetime, dst.ty, dst.mutability),
                    assume,
                ),
            }
        }

        let predicate = obligation.predicate.skip_binder();

        let mut assume = predicate.trait_ref.args.const_at(2);
        // FIXME(mgca): We should shallowly normalize this.
        if self.tcx().features().generic_const_exprs() {
            assume = crate::traits::evaluate_const(self.infcx, assume, obligation.param_env)
        }
        let Some(assume) = rustc_transmute::Assume::from_const(self.infcx.tcx, assume) else {
            return Err(Unimplemented);
        };

        let dst = predicate.trait_ref.args.type_at(0);
        let src = predicate.trait_ref.args.type_at(1);

        debug!(?src, ?dst);
        let mut transmute_env = rustc_transmute::TransmuteTypeEnv::new(self.infcx.tcx);
        let maybe_transmutable =
            transmute_env.is_transmutable(rustc_transmute::Types { dst, src }, assume);

        let fully_flattened = match maybe_transmutable {
            Answer::No(_) => Err(Unimplemented)?,
            Answer::If(cond) => flatten_answer_tree(self.tcx(), obligation, cond, assume),
            Answer::Yes => PredicateObligations::new(),
        };

        debug!(?fully_flattened);
        Ok(fully_flattened)
    }

    /// This handles the case where an `auto trait Foo` impl is being used.
    /// The idea is that the impl applies to `X : Foo` if the following conditions are met:
    ///
    /// 1. For each constituent type `Y` in `X`, `Y : Foo` holds
    /// 2. For each where-clause `C` declared on `Foo`, `[Self => X] C` holds.
    fn confirm_auto_impl_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        ensure_sufficient_stack(|| {
            assert_eq!(obligation.predicate.polarity(), ty::PredicatePolarity::Positive);

            let self_ty =
                obligation.predicate.self_ty().map_bound(|ty| self.infcx.shallow_resolve(ty));

            let types = self.constituent_types_for_ty(self_ty)?;
            let types = self.infcx.enter_forall_and_leak_universe(types);

            let cause = obligation.derived_cause(ObligationCauseCode::BuiltinDerived);
            let obligations = self.collect_predicates_for_types(
                obligation.param_env,
                cause,
                obligation.recursion_depth + 1,
                obligation.predicate.def_id(),
                types,
            );

            Ok(obligations)
        })
    }

    fn confirm_impl_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        impl_def_id: DefId,
    ) -> ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>> {
        debug!(?obligation, ?impl_def_id, "confirm_impl_candidate");

        // First, create the generic parameters by matching the impl again,
        // this time not in a probe.
        let args = self.rematch_impl(impl_def_id, obligation);
        debug!(?args, "impl args");
        ensure_sufficient_stack(|| {
            self.vtable_impl(
                impl_def_id,
                args,
                &obligation.cause,
                obligation.recursion_depth + 1,
                obligation.param_env,
                obligation.predicate,
            )
        })
    }

    fn vtable_impl(
        &mut self,
        impl_def_id: DefId,
        args: Normalized<'tcx, GenericArgsRef<'tcx>>,
        cause: &ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
        parent_trait_pred: ty::Binder<'tcx, ty::TraitPredicate<'tcx>>,
    ) -> ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>> {
        debug!(?impl_def_id, ?args, ?recursion_depth, "vtable_impl");

        let mut impl_obligations = self.impl_or_trait_obligations(
            cause,
            recursion_depth,
            param_env,
            impl_def_id,
            args.value,
            parent_trait_pred,
        );

        debug!(?impl_obligations, "vtable_impl");

        // Because of RFC447, the impl-trait-ref and obligations
        // are sufficient to determine the impl args, without
        // relying on projections in the impl-trait-ref.
        //
        // e.g., `impl<U: Tr, V: Iterator<Item=U>> Foo<<U as Tr>::T> for V`
        impl_obligations.extend(args.obligations);

        ImplSourceUserDefinedData { impl_def_id, args: args.value, nested: impl_obligations }
    }

    fn confirm_object_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        index: usize,
    ) -> Result<ImplSource<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let tcx = self.tcx();
        debug!(?obligation, ?index, "confirm_object_candidate");

        let trait_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(trait_predicate.self_ty());
        let ty::Dynamic(data, ..) = *self_ty.kind() else {
            span_bug!(obligation.cause.span, "object candidate with non-object");
        };

        let object_trait_ref = data.principal().unwrap_or_else(|| {
            span_bug!(obligation.cause.span, "object candidate with no principal")
        });
        let object_trait_ref = self.infcx.instantiate_binder_with_fresh_vars(
            obligation.cause.span,
            HigherRankedType,
            object_trait_ref,
        );
        let object_trait_ref = object_trait_ref.with_self_ty(self.tcx(), self_ty);

        let mut nested = PredicateObligations::new();

        let mut supertraits = util::supertraits(tcx, ty::Binder::dummy(object_trait_ref));
        let unnormalized_upcast_trait_ref =
            supertraits.nth(index).expect("supertraits iterator no longer has as many elements");

        let upcast_trait_ref = self.infcx.instantiate_binder_with_fresh_vars(
            obligation.cause.span,
            HigherRankedType,
            unnormalized_upcast_trait_ref,
        );
        let upcast_trait_ref = normalize_with_depth_to(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            upcast_trait_ref,
            &mut nested,
        );

        nested.extend(
            self.infcx
                .at(&obligation.cause, obligation.param_env)
                .eq(DefineOpaqueTypes::No, trait_predicate.trait_ref, upcast_trait_ref)
                .map(|InferOk { obligations, .. }| obligations)
                .map_err(|_| Unimplemented)?,
        );

        // Check supertraits hold. This is so that their associated type bounds
        // will be checked in the code below.
        for (supertrait, _) in tcx
            .explicit_super_predicates_of(trait_predicate.def_id())
            .iter_instantiated_copied(tcx, trait_predicate.trait_ref.args)
        {
            let normalized_supertrait = normalize_with_depth_to(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                supertrait,
                &mut nested,
            );
            nested.push(obligation.with(tcx, normalized_supertrait));
        }

        let assoc_types: Vec<_> = tcx
            .associated_items(trait_predicate.def_id())
            .in_definition_order()
            // Associated types that require `Self: Sized` do not show up in the built-in
            // implementation of `Trait for dyn Trait`, and can be dropped here.
            .filter(|item| !tcx.generics_require_sized_self(item.def_id))
            .filter_map(
                |item| if item.kind == ty::AssocKind::Type { Some(item.def_id) } else { None },
            )
            .collect();

        for assoc_type in assoc_types {
            let defs: &ty::Generics = tcx.generics_of(assoc_type);

            if !defs.own_params.is_empty() {
                tcx.dcx().span_delayed_bug(
                    obligation.cause.span,
                    "GATs in trait object shouldn't have been considered",
                );
                return Err(SelectionError::TraitDynIncompatible(trait_predicate.trait_ref.def_id));
            }

            // This maybe belongs in wf, but that can't (doesn't) handle
            // higher-ranked things.
            // Prevent, e.g., `dyn Iterator<Item = str>`.
            for bound in self.tcx().item_bounds(assoc_type).transpose_iter() {
                let normalized_bound = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    bound.instantiate(tcx, trait_predicate.trait_ref.args),
                    &mut nested,
                );
                nested.push(obligation.with(tcx, normalized_bound));
            }
        }

        debug!(?nested, "object nested obligations");

        Ok(ImplSource::Builtin(BuiltinImplSource::Object(index), nested))
    }

    fn confirm_fn_pointer_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        debug!(?obligation, "confirm_fn_pointer_candidate");
        let placeholder_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(placeholder_predicate.self_ty());

        let tcx = self.tcx();
        let sig = self_ty.fn_sig(tcx);
        let trait_ref = closure_trait_ref_and_return_type(
            tcx,
            obligation.predicate.def_id(),
            self_ty,
            sig,
            util::TupleArgumentsFlag::Yes,
        )
        .map_bound(|(trait_ref, _)| trait_ref);

        let mut nested =
            self.equate_trait_refs(obligation.with(tcx, placeholder_predicate), trait_ref)?;
        let cause = obligation.derived_cause(ObligationCauseCode::BuiltinDerived);

        // Confirm the `type Output: Sized;` bound that is present on `FnOnce`
        let output_ty = self.infcx.enter_forall_and_leak_universe(sig.output());
        let output_ty = normalize_with_depth_to(
            self,
            obligation.param_env,
            cause.clone(),
            obligation.recursion_depth,
            output_ty,
            &mut nested,
        );
        let tr = ty::TraitRef::new(
            self.tcx(),
            self.tcx().require_lang_item(LangItem::Sized, Some(cause.span)),
            [output_ty],
        );
        nested.push(Obligation::new(self.infcx.tcx, cause, obligation.param_env, tr));

        Ok(nested)
    }

    fn confirm_trait_alias_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> PredicateObligations<'tcx> {
        debug!(?obligation, "confirm_trait_alias_candidate");

        let predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let trait_ref = predicate.trait_ref;
        let trait_def_id = trait_ref.def_id;
        let args = trait_ref.args;

        let trait_obligations = self.impl_or_trait_obligations(
            &obligation.cause,
            obligation.recursion_depth,
            obligation.param_env,
            trait_def_id,
            args,
            obligation.predicate,
        );

        debug!(?trait_def_id, ?trait_obligations, "trait alias obligations");

        trait_obligations
    }

    fn confirm_coroutine_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let placeholder_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(placeholder_predicate.self_ty());
        let ty::Coroutine(coroutine_def_id, args) = *self_ty.kind() else {
            bug!("closure candidate for non-closure {:?}", obligation);
        };

        debug!(?obligation, ?coroutine_def_id, ?args, "confirm_coroutine_candidate");

        let coroutine_sig = args.as_coroutine().sig();

        let (trait_ref, _, _) = super::util::coroutine_trait_ref_and_outputs(
            self.tcx(),
            obligation.predicate.def_id(),
            self_ty,
            coroutine_sig,
        );

        let nested = self.equate_trait_refs(
            obligation.with(self.tcx(), placeholder_predicate),
            ty::Binder::dummy(trait_ref),
        )?;
        debug!(?trait_ref, ?nested, "coroutine candidate obligations");

        Ok(nested)
    }

    fn confirm_future_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let placeholder_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(placeholder_predicate.self_ty());
        let ty::Coroutine(coroutine_def_id, args) = *self_ty.kind() else {
            bug!("closure candidate for non-closure {:?}", obligation);
        };

        debug!(?obligation, ?coroutine_def_id, ?args, "confirm_future_candidate");

        let coroutine_sig = args.as_coroutine().sig();

        let (trait_ref, _) = super::util::future_trait_ref_and_outputs(
            self.tcx(),
            obligation.predicate.def_id(),
            self_ty,
            coroutine_sig,
        );

        let nested = self.equate_trait_refs(
            obligation.with(self.tcx(), placeholder_predicate),
            ty::Binder::dummy(trait_ref),
        )?;
        debug!(?trait_ref, ?nested, "future candidate obligations");

        Ok(nested)
    }

    fn confirm_iterator_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let placeholder_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(placeholder_predicate.self_ty());
        let ty::Coroutine(coroutine_def_id, args) = *self_ty.kind() else {
            bug!("closure candidate for non-closure {:?}", obligation);
        };

        debug!(?obligation, ?coroutine_def_id, ?args, "confirm_iterator_candidate");

        let gen_sig = args.as_coroutine().sig();

        let (trait_ref, _) = super::util::iterator_trait_ref_and_outputs(
            self.tcx(),
            obligation.predicate.def_id(),
            self_ty,
            gen_sig,
        );

        let nested = self.equate_trait_refs(
            obligation.with(self.tcx(), placeholder_predicate),
            ty::Binder::dummy(trait_ref),
        )?;
        debug!(?trait_ref, ?nested, "iterator candidate obligations");

        Ok(nested)
    }

    fn confirm_async_iterator_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let placeholder_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(placeholder_predicate.self_ty());
        let ty::Coroutine(coroutine_def_id, args) = *self_ty.kind() else {
            bug!("closure candidate for non-closure {:?}", obligation);
        };

        debug!(?obligation, ?coroutine_def_id, ?args, "confirm_async_iterator_candidate");

        let gen_sig = args.as_coroutine().sig();

        let (trait_ref, _) = super::util::async_iterator_trait_ref_and_outputs(
            self.tcx(),
            obligation.predicate.def_id(),
            self_ty,
            gen_sig,
        );

        let nested = self.equate_trait_refs(
            obligation.with(self.tcx(), placeholder_predicate),
            ty::Binder::dummy(trait_ref),
        )?;
        debug!(?trait_ref, ?nested, "iterator candidate obligations");

        Ok(nested)
    }

    #[instrument(skip(self), level = "debug")]
    fn confirm_closure_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let placeholder_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty: Ty<'_> = self.infcx.shallow_resolve(placeholder_predicate.self_ty());

        let trait_ref = match *self_ty.kind() {
            ty::Closure(..) => {
                self.closure_trait_ref_unnormalized(self_ty, obligation.predicate.def_id())
            }
            ty::CoroutineClosure(_, args) => {
                args.as_coroutine_closure().coroutine_closure_sig().map_bound(|sig| {
                    ty::TraitRef::new(
                        self.tcx(),
                        obligation.predicate.def_id(),
                        [self_ty, sig.tupled_inputs_ty],
                    )
                })
            }
            _ => {
                bug!("closure candidate for non-closure {:?}", obligation);
            }
        };

        self.equate_trait_refs(obligation.with(self.tcx(), placeholder_predicate), trait_ref)
    }

    #[instrument(skip(self), level = "debug")]
    fn confirm_async_closure_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let placeholder_predicate = self.infcx.enter_forall_and_leak_universe(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(placeholder_predicate.self_ty());

        let tcx = self.tcx();

        let mut nested = PredicateObligations::new();
        let (trait_ref, kind_ty) = match *self_ty.kind() {
            ty::CoroutineClosure(_, args) => {
                let args = args.as_coroutine_closure();
                let trait_ref = args.coroutine_closure_sig().map_bound(|sig| {
                    ty::TraitRef::new(
                        self.tcx(),
                        obligation.predicate.def_id(),
                        [self_ty, sig.tupled_inputs_ty],
                    )
                });

                // Note that unlike below, we don't need to check `Future + Sized` for
                // the output coroutine because they are `Future + Sized` by construction.

                (trait_ref, args.kind_ty())
            }
            ty::FnDef(..) | ty::FnPtr(..) => {
                let sig = self_ty.fn_sig(tcx);
                let trait_ref = sig.map_bound(|sig| {
                    ty::TraitRef::new(
                        self.tcx(),
                        obligation.predicate.def_id(),
                        [self_ty, Ty::new_tup(tcx, sig.inputs())],
                    )
                });

                // We must additionally check that the return type impls `Future + Sized`.
                let future_trait_def_id = tcx.require_lang_item(LangItem::Future, None);
                nested.push(obligation.with(
                    tcx,
                    sig.output().map_bound(|output_ty| {
                        ty::TraitRef::new(tcx, future_trait_def_id, [output_ty])
                    }),
                ));
                let sized_trait_def_id = tcx.require_lang_item(LangItem::Sized, None);
                nested.push(obligation.with(
                    tcx,
                    sig.output().map_bound(|output_ty| {
                        ty::TraitRef::new(tcx, sized_trait_def_id, [output_ty])
                    }),
                ));

                (trait_ref, Ty::from_closure_kind(tcx, ty::ClosureKind::Fn))
            }
            ty::Closure(_, args) => {
                let args = args.as_closure();
                let sig = args.sig();
                let trait_ref = sig.map_bound(|sig| {
                    ty::TraitRef::new(
                        self.tcx(),
                        obligation.predicate.def_id(),
                        [self_ty, sig.inputs()[0]],
                    )
                });

                // We must additionally check that the return type impls `Future + Sized`.
                let future_trait_def_id = tcx.require_lang_item(LangItem::Future, None);
                let placeholder_output_ty = self.infcx.enter_forall_and_leak_universe(sig.output());
                nested.push(obligation.with(
                    tcx,
                    ty::TraitRef::new(tcx, future_trait_def_id, [placeholder_output_ty]),
                ));
                let sized_trait_def_id = tcx.require_lang_item(LangItem::Sized, None);
                nested.push(obligation.with(
                    tcx,
                    sig.output().map_bound(|output_ty| {
                        ty::TraitRef::new(tcx, sized_trait_def_id, [output_ty])
                    }),
                ));

                (trait_ref, args.kind_ty())
            }
            _ => bug!("expected callable type for AsyncFn candidate"),
        };

        nested.extend(
            self.equate_trait_refs(obligation.with(tcx, placeholder_predicate), trait_ref)?,
        );

        let goal_kind =
            self.tcx().async_fn_trait_kind_from_def_id(obligation.predicate.def_id()).unwrap();

        // If we have not yet determiend the `ClosureKind` of the closure or coroutine-closure,
        // then additionally register an `AsyncFnKindHelper` goal which will fail if the kind
        // is constrained to an insufficient type later on.
        if let Some(closure_kind) = self.infcx.shallow_resolve(kind_ty).to_opt_closure_kind() {
            if !closure_kind.extends(goal_kind) {
                return Err(SelectionError::Unimplemented);
            }
        } else {
            nested.push(Obligation::new(
                self.tcx(),
                obligation.derived_cause(ObligationCauseCode::BuiltinDerived),
                obligation.param_env,
                ty::TraitRef::new(
                    self.tcx(),
                    self.tcx().require_lang_item(
                        LangItem::AsyncFnKindHelper,
                        Some(obligation.cause.span),
                    ),
                    [kind_ty, Ty::from_closure_kind(self.tcx(), goal_kind)],
                ),
            ));
        }

        Ok(nested)
    }

    /// In the case of closure types and fn pointers,
    /// we currently treat the input type parameters on the trait as
    /// outputs. This means that when we have a match we have only
    /// considered the self type, so we have to go back and make sure
    /// to relate the argument types too. This is kind of wrong, but
    /// since we control the full set of impls, also not that wrong,
    /// and it DOES yield better error messages (since we don't report
    /// errors as if there is no applicable impl, but rather report
    /// errors are about mismatched argument types.
    ///
    /// Here is an example. Imagine we have a closure expression
    /// and we desugared it so that the type of the expression is
    /// `Closure`, and `Closure` expects `i32` as argument. Then it
    /// is "as if" the compiler generated this impl:
    /// ```ignore (illustrative)
    /// impl Fn(i32) for Closure { ... }
    /// ```
    /// Now imagine our obligation is `Closure: Fn(usize)`. So far
    /// we have matched the self type `Closure`. At this point we'll
    /// compare the `i32` to `usize` and generate an error.
    ///
    /// Note that this checking occurs *after* the impl has selected,
    /// because these output type parameters should not affect the
    /// selection of the impl. Therefore, if there is a mismatch, we
    /// report an error to the user.
    #[instrument(skip(self), level = "trace")]
    fn equate_trait_refs(
        &mut self,
        obligation: TraitObligation<'tcx>,
        found_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<PredicateObligations<'tcx>, SelectionError<'tcx>> {
        let found_trait_ref = self.infcx.instantiate_binder_with_fresh_vars(
            obligation.cause.span,
            HigherRankedType,
            found_trait_ref,
        );
        // Normalize the obligation and expected trait refs together, because why not
        let Normalized { obligations: nested, value: (obligation_trait_ref, found_trait_ref) } =
            ensure_sufficient_stack(|| {
                normalize_with_depth(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    (obligation.predicate.trait_ref, found_trait_ref),
                )
            });

        // needed to define opaque types for tests/ui/type-alias-impl-trait/assoc-projection-ice.rs
        self.infcx
            .at(&obligation.cause, obligation.param_env)
            .eq(DefineOpaqueTypes::Yes, obligation_trait_ref, found_trait_ref)
            .map(|InferOk { mut obligations, .. }| {
                obligations.extend(nested);
                obligations
            })
            .map_err(|terr| {
                SignatureMismatch(Box::new(SignatureMismatchData {
                    expected_trait_ref: obligation_trait_ref,
                    found_trait_ref,
                    terr,
                }))
            })
    }

    fn confirm_trait_upcasting_unsize_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
        idx: usize,
    ) -> Result<ImplSource<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let tcx = self.tcx();

        // `assemble_candidates_for_unsizing` should ensure there are no late-bound
        // regions here. See the comment there for more details.
        let predicate = obligation.predicate.no_bound_vars().unwrap();
        let a_ty = self.infcx.shallow_resolve(predicate.self_ty());
        let b_ty = self.infcx.shallow_resolve(predicate.trait_ref.args.type_at(1));

        let ty::Dynamic(a_data, a_region, ty::Dyn) = *a_ty.kind() else {
            bug!("expected `dyn` type in `confirm_trait_upcasting_unsize_candidate`")
        };
        let ty::Dynamic(b_data, b_region, ty::Dyn) = *b_ty.kind() else {
            bug!("expected `dyn` type in `confirm_trait_upcasting_unsize_candidate`")
        };

        let source_principal = a_data.principal().unwrap().with_self_ty(tcx, a_ty);
        let unnormalized_upcast_principal =
            util::supertraits(tcx, source_principal).nth(idx).unwrap();

        let nested = self
            .match_upcast_principal(
                obligation,
                unnormalized_upcast_principal,
                a_data,
                b_data,
                a_region,
                b_region,
            )?
            .expect("did not expect ambiguity during confirmation");

        Ok(ImplSource::Builtin(BuiltinImplSource::TraitUpcasting(idx), nested))
    }

    fn confirm_builtin_unsize_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> Result<ImplSource<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let tcx = self.tcx();

        // `assemble_candidates_for_unsizing` should ensure there are no late-bound
        // regions here. See the comment there for more details.
        let source = self.infcx.shallow_resolve(obligation.self_ty().no_bound_vars().unwrap());
        let target = obligation.predicate.skip_binder().trait_ref.args.type_at(1);
        let target = self.infcx.shallow_resolve(target);
        debug!(?source, ?target, "confirm_builtin_unsize_candidate");

        Ok(match (source.kind(), target.kind()) {
            // Trait+Kx+'a -> Trait+Ky+'b (auto traits and lifetime subtyping).
            (&ty::Dynamic(data_a, r_a, dyn_a), &ty::Dynamic(data_b, r_b, dyn_b))
                if dyn_a == dyn_b =>
            {
                // See `assemble_candidates_for_unsizing` for more info.
                // We already checked the compatibility of auto traits within `assemble_candidates_for_unsizing`.
                let existential_predicates = if data_b.principal().is_some() {
                    tcx.mk_poly_existential_predicates_from_iter(
                        data_a
                            .principal()
                            .map(|b| b.map_bound(ty::ExistentialPredicate::Trait))
                            .into_iter()
                            .chain(
                                data_a
                                    .projection_bounds()
                                    .map(|b| b.map_bound(ty::ExistentialPredicate::Projection)),
                            )
                            .chain(
                                data_b
                                    .auto_traits()
                                    .map(ty::ExistentialPredicate::AutoTrait)
                                    .map(ty::Binder::dummy),
                            ),
                    )
                } else {
                    // If we're unsizing to a dyn type that has no principal, then drop
                    // the principal and projections from the type. We use the auto traits
                    // from the RHS type since as we noted that we've checked for auto
                    // trait compatibility during unsizing.
                    tcx.mk_poly_existential_predicates_from_iter(
                        data_b
                            .auto_traits()
                            .map(ty::ExistentialPredicate::AutoTrait)
                            .map(ty::Binder::dummy),
                    )
                };
                let source_trait = Ty::new_dynamic(tcx, existential_predicates, r_b, dyn_a);

                // Require that the traits involved in this upcast are **equal**;
                // only the **lifetime bound** is changed.
                let InferOk { mut obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .sup(DefineOpaqueTypes::Yes, target, source_trait)
                    .map_err(|_| Unimplemented)?;

                // Register one obligation for 'a: 'b.
                let outlives = ty::OutlivesPredicate(r_a, r_b);
                obligations.push(Obligation::with_depth(
                    tcx,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    obligation.predicate.rebind(outlives),
                ));

                ImplSource::Builtin(BuiltinImplSource::Misc, obligations)
            }

            // `T` -> `dyn Trait`
            (_, &ty::Dynamic(data, r, ty::Dyn)) => {
                let mut object_dids = data.auto_traits().chain(data.principal_def_id());
                if let Some(did) = object_dids.find(|did| !tcx.is_dyn_compatible(*did)) {
                    return Err(TraitDynIncompatible(did));
                }

                let predicate_to_obligation = |predicate| {
                    Obligation::with_depth(
                        tcx,
                        obligation.cause.clone(),
                        obligation.recursion_depth + 1,
                        obligation.param_env,
                        predicate,
                    )
                };

                // Create obligations:
                //  - Casting `T` to `Trait`
                //  - For all the various builtin bounds attached to the object cast. (In other
                //  words, if the object type is `Foo + Send`, this would create an obligation for
                //  the `Send` check.)
                //  - Projection predicates
                let mut nested: PredicateObligations<'_> = data
                    .iter()
                    .map(|predicate| predicate_to_obligation(predicate.with_self_ty(tcx, source)))
                    .collect();

                // We can only make objects from sized types.
                let tr = ty::TraitRef::new(
                    tcx,
                    tcx.require_lang_item(LangItem::Sized, Some(obligation.cause.span)),
                    [source],
                );
                nested.push(predicate_to_obligation(tr.upcast(tcx)));

                // If the type is `Foo + 'a`, ensure that the type
                // being cast to `Foo + 'a` outlives `'a`:
                let outlives = ty::OutlivesPredicate(source, r);
                nested.push(predicate_to_obligation(
                    ty::ClauseKind::TypeOutlives(outlives).upcast(tcx),
                ));

                // Require that all AFIT will return something that can be coerced into `dyn*`
                // -- a shim will be responsible for doing the actual coercion to `dyn*`.
                if let Some(principal) = data.principal() {
                    for supertrait in
                        elaborate::supertraits(tcx, principal.with_self_ty(tcx, source))
                    {
                        if tcx.is_trait_alias(supertrait.def_id()) {
                            continue;
                        }

                        for &assoc_item in tcx.associated_item_def_ids(supertrait.def_id()) {
                            if !tcx.is_impl_trait_in_trait(assoc_item) {
                                continue;
                            }

                            // RPITITs with `Self: Sized` don't need to be checked.
                            if tcx.generics_require_sized_self(assoc_item) {
                                continue;
                            }

                            let pointer_like_goal = pointer_like_goal_for_rpitit(
                                tcx,
                                supertrait,
                                assoc_item,
                                &obligation.cause,
                            );

                            nested.push(predicate_to_obligation(pointer_like_goal.upcast(tcx)));
                        }
                    }
                }

                ImplSource::Builtin(BuiltinImplSource::Misc, nested)
            }

            // `[T; n]` -> `[T]`
            (&ty::Array(a, _), &ty::Slice(b)) => {
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(DefineOpaqueTypes::Yes, b, a)
                    .map_err(|_| Unimplemented)?;

                ImplSource::Builtin(BuiltinImplSource::Misc, obligations)
            }

            // `Struct<T>` -> `Struct<U>`
            (&ty::Adt(def, args_a), &ty::Adt(_, args_b)) => {
                let unsizing_params = tcx.unsizing_params_for_adt(def.did());
                if unsizing_params.is_empty() {
                    return Err(Unimplemented);
                }

                let tail_field = def.non_enum_variant().tail();
                let tail_field_ty = tcx.type_of(tail_field.did);

                let mut nested = PredicateObligations::new();

                // Extract `TailField<T>` and `TailField<U>` from `Struct<T>` and `Struct<U>`,
                // normalizing in the process, since `type_of` returns something directly from
                // HIR ty lowering (which means it's un-normalized).
                let source_tail = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    tail_field_ty.instantiate(tcx, args_a),
                    &mut nested,
                );
                let target_tail = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    tail_field_ty.instantiate(tcx, args_b),
                    &mut nested,
                );

                // Check that the source struct with the target's
                // unsizing parameters is equal to the target.
                let args =
                    tcx.mk_args_from_iter(args_a.iter().enumerate().map(|(i, k)| {
                        if unsizing_params.contains(i as u32) { args_b[i] } else { k }
                    }));
                let new_struct = Ty::new_adt(tcx, def, args);
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(DefineOpaqueTypes::Yes, target, new_struct)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Construct the nested `TailField<T>: Unsize<TailField<U>>` predicate.
                let tail_unsize_obligation = obligation.with(
                    tcx,
                    ty::TraitRef::new(
                        tcx,
                        obligation.predicate.def_id(),
                        [source_tail, target_tail],
                    ),
                );
                nested.push(tail_unsize_obligation);

                ImplSource::Builtin(BuiltinImplSource::Misc, nested)
            }

            _ => bug!("source: {source}, target: {target}"),
        })
    }

    fn confirm_bikeshed_guaranteed_no_drop_candidate(
        &mut self,
        obligation: &PolyTraitObligation<'tcx>,
    ) -> ImplSource<'tcx, PredicateObligation<'tcx>> {
        let mut obligations = thin_vec![];

        let tcx = self.tcx();
        let self_ty = obligation.predicate.self_ty();
        match *self_ty.skip_binder().kind() {
            // `&mut T` and `&T` always implement `BikeshedGuaranteedNoDrop`.
            ty::Ref(..) => {}
            // `ManuallyDrop<T>` always implements `BikeshedGuaranteedNoDrop`.
            ty::Adt(def, _) if def.is_manually_drop() => {}
            // Arrays and tuples implement `BikeshedGuaranteedNoDrop` only if
            // their constituent types implement `BikeshedGuaranteedNoDrop`.
            ty::Tuple(tys) => {
                obligations.extend(tys.iter().map(|elem_ty| {
                    obligation.with(
                        tcx,
                        self_ty.rebind(ty::TraitRef::new(
                            tcx,
                            obligation.predicate.def_id(),
                            [elem_ty],
                        )),
                    )
                }));
            }
            ty::Array(elem_ty, _) => {
                obligations.push(obligation.with(
                    tcx,
                    self_ty.rebind(ty::TraitRef::new(
                        tcx,
                        obligation.predicate.def_id(),
                        [elem_ty],
                    )),
                ));
            }

            // All other types implement `BikeshedGuaranteedNoDrop` only if
            // they implement `Copy`. We could be smart here and short-circuit
            // some trivially `Copy`/`!Copy` types, but there's no benefit.
            ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Error(_)
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Bool
            | ty::Float(_)
            | ty::Char
            | ty::RawPtr(..)
            | ty::Never
            | ty::Pat(..)
            | ty::Dynamic(..)
            | ty::Str
            | ty::Slice(_)
            | ty::Foreign(..)
            | ty::Adt(..)
            | ty::Alias(..)
            | ty::Param(_)
            | ty::Placeholder(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::UnsafeBinder(_)
            | ty::CoroutineWitness(..)
            | ty::Bound(..) => {
                obligations.push(obligation.with(
                    tcx,
                    self_ty.map_bound(|ty| {
                        ty::TraitRef::new(
                            tcx,
                            tcx.require_lang_item(LangItem::Copy, Some(obligation.cause.span)),
                            [ty],
                        )
                    }),
                ));
            }

            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                panic!("unexpected type `{self_ty:?}`")
            }
        }

        ImplSource::Builtin(BuiltinImplSource::Misc, obligations)
    }
}

/// Compute a goal that some RPITIT (right now, only RPITITs corresponding to Futures)
/// implements the `PointerLike` trait, which is a requirement for the RPITIT to be
/// coercible to `dyn* Future`, which is itself a requirement for the RPITIT's parent
/// trait to be coercible to `dyn Trait`.
///
/// We do this given a supertrait's substitutions, and then augment the substitutions
/// with bound variables to compute the goal universally. Given that `PointerLike` has
/// no region requirements (at least for the built-in pointer types), this shouldn't
/// *really* matter, but it is the best choice for soundness.
fn pointer_like_goal_for_rpitit<'tcx>(
    tcx: TyCtxt<'tcx>,
    supertrait: ty::PolyTraitRef<'tcx>,
    rpitit_item: DefId,
    cause: &ObligationCause<'tcx>,
) -> ty::PolyTraitRef<'tcx> {
    let mut bound_vars = supertrait.bound_vars().to_vec();

    let args = supertrait.skip_binder().args.extend_to(tcx, rpitit_item, |arg, _| match arg.kind {
        ty::GenericParamDefKind::Lifetime => {
            let kind = ty::BoundRegionKind::Named(arg.def_id, tcx.item_name(arg.def_id));
            bound_vars.push(ty::BoundVariableKind::Region(kind));
            ty::Region::new_bound(
                tcx,
                ty::INNERMOST,
                ty::BoundRegion { var: ty::BoundVar::from_usize(bound_vars.len() - 1), kind },
            )
            .into()
        }
        ty::GenericParamDefKind::Type { .. } | ty::GenericParamDefKind::Const { .. } => {
            unreachable!()
        }
    });

    ty::Binder::bind_with_vars(
        ty::TraitRef::new(
            tcx,
            tcx.require_lang_item(LangItem::PointerLike, Some(cause.span)),
            [Ty::new_projection_from_args(tcx, rpitit_item, args)],
        ),
        tcx.mk_bound_variable_kinds(&bound_vars),
    )
}
