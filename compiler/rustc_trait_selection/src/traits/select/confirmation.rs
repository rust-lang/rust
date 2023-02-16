//! Confirmation.
//!
//! Confirmation unifies the output type parameters of the trait
//! with the values found in the obligation, possibly yielding a
//! type error. See the [rustc dev guide] for more details.
//!
//! [rustc dev guide]:
//! https://rustc-dev-guide.rust-lang.org/traits/resolution.html#confirmation
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::InferOk;
use rustc_infer::infer::LateBoundRegionConversionTime::HigherRankedType;
use rustc_middle::ty::{
    self, Binder, GenericParamDefKind, InternalSubsts, SubstsRef, ToPolyTraitRef, ToPredicate,
    TraitRef, Ty, TyCtxt, TypeVisitable,
};
use rustc_session::config::TraitSolver;
use rustc_span::def_id::DefId;

use crate::traits::project::{normalize_with_depth, normalize_with_depth_to};
use crate::traits::util::{self, closure_trait_ref_and_return_type, predicate_for_trait_def};
use crate::traits::vtable::{
    count_own_vtable_entries, prepare_vtable_segments, vtable_trait_first_method_offset,
    VtblSegment,
};
use crate::traits::{
    BuiltinDerivedObligation, ImplDerivedObligation, ImplDerivedObligationCause, ImplSource,
    ImplSourceAutoImplData, ImplSourceBuiltinData, ImplSourceClosureData,
    ImplSourceConstDestructData, ImplSourceFnPointerData, ImplSourceFutureData,
    ImplSourceGeneratorData, ImplSourceObjectData, ImplSourceTraitAliasData,
    ImplSourceTraitUpcastingData, ImplSourceUserDefinedData, Normalized, ObjectCastObligation,
    Obligation, ObligationCause, OutputTypeParameterMismatch, PredicateObligation, Selection,
    SelectionError, TraitNotObjectSafe, TraitObligation, Unimplemented,
};

use super::BuiltinImplConditions;
use super::SelectionCandidate::{self, *};
use super::SelectionContext;

use std::iter;
use std::ops::ControlFlow;

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    pub(super) fn confirm_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidate: SelectionCandidate<'tcx>,
    ) -> Result<Selection<'tcx>, SelectionError<'tcx>> {
        let mut impl_src = match candidate {
            BuiltinCandidate { has_nested } => {
                let data = self.confirm_builtin_candidate(obligation, has_nested);
                ImplSource::Builtin(data)
            }

            TransmutabilityCandidate => {
                let data = self.confirm_transmutability_candidate(obligation)?;
                ImplSource::Builtin(data)
            }

            ParamCandidate(param) => {
                let obligations =
                    self.confirm_param_candidate(obligation, param.map_bound(|t| t.trait_ref));
                ImplSource::Param(obligations, param.skip_binder().constness)
            }

            ImplCandidate(impl_def_id) => {
                ImplSource::UserDefined(self.confirm_impl_candidate(obligation, impl_def_id))
            }

            AutoImplCandidate => {
                let data = self.confirm_auto_impl_candidate(obligation);
                ImplSource::AutoImpl(data)
            }

            ProjectionCandidate(idx, constness) => {
                let obligations = self.confirm_projection_candidate(obligation, idx)?;
                ImplSource::Param(obligations, constness)
            }

            ObjectCandidate(idx) => {
                let data = self.confirm_object_candidate(obligation, idx)?;
                ImplSource::Object(data)
            }

            ClosureCandidate { .. } => {
                let vtable_closure = self.confirm_closure_candidate(obligation)?;
                ImplSource::Closure(vtable_closure)
            }

            GeneratorCandidate => {
                let vtable_generator = self.confirm_generator_candidate(obligation)?;
                ImplSource::Generator(vtable_generator)
            }

            FutureCandidate => {
                let vtable_future = self.confirm_future_candidate(obligation)?;
                ImplSource::Future(vtable_future)
            }

            FnPointerCandidate { is_const } => {
                let data = self.confirm_fn_pointer_candidate(obligation, is_const)?;
                ImplSource::FnPointer(data)
            }

            TraitAliasCandidate => {
                let data = self.confirm_trait_alias_candidate(obligation);
                ImplSource::TraitAlias(data)
            }

            BuiltinObjectCandidate => {
                // This indicates something like `Trait + Send: Send`. In this case, we know that
                // this holds because that's what the object type is telling us, and there's really
                // no additional obligations to prove and no types in particular to unify, etc.
                ImplSource::Param(Vec::new(), ty::BoundConstness::NotConst)
            }

            BuiltinUnsizeCandidate => {
                let data = self.confirm_builtin_unsize_candidate(obligation)?;
                ImplSource::Builtin(data)
            }

            TraitUpcastingUnsizeCandidate(idx) => {
                let data = self.confirm_trait_upcasting_unsize_candidate(obligation, idx)?;
                ImplSource::TraitUpcasting(data)
            }

            ConstDestructCandidate(def_id) => {
                let data = self.confirm_const_destruct_candidate(obligation, def_id)?;
                ImplSource::ConstDestruct(data)
            }
        };

        if !obligation.predicate.is_const_if_const() {
            // normalize nested predicates according to parent predicate's constness.
            impl_src = impl_src.map(|mut o| {
                o.predicate = o.predicate.without_const(self.tcx());
                o
            });
        }

        Ok(impl_src)
    }

    fn confirm_projection_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        idx: usize,
    ) -> Result<Vec<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let tcx = self.tcx();

        let trait_predicate = self.infcx.shallow_resolve(obligation.predicate);
        let placeholder_trait_predicate =
            self.infcx.instantiate_binder_with_placeholders(trait_predicate).trait_ref;
        let placeholder_self_ty = placeholder_trait_predicate.self_ty();
        let placeholder_trait_predicate = ty::Binder::dummy(placeholder_trait_predicate);
        let (def_id, substs) = match *placeholder_self_ty.kind() {
            ty::Alias(_, ty::AliasTy { def_id, substs, .. }) => (def_id, substs),
            _ => bug!("projection candidate for unexpected type: {:?}", placeholder_self_ty),
        };

        let candidate_predicate = tcx.item_bounds(def_id).map_bound(|i| i[idx]).subst(tcx, substs);
        let candidate = candidate_predicate
            .to_opt_poly_trait_pred()
            .expect("projection candidate is not a trait predicate")
            .map_bound(|t| t.trait_ref);
        let mut obligations = Vec::new();
        let candidate = normalize_with_depth_to(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            candidate,
            &mut obligations,
        );

        obligations.extend(self.infcx.commit_if_ok(|_| {
            self.infcx
                .at(&obligation.cause, obligation.param_env)
                .sup(placeholder_trait_predicate, candidate)
                .map(|InferOk { obligations, .. }| obligations)
                .map_err(|_| Unimplemented)
        })?);

        if let ty::Alias(ty::Projection, ..) = placeholder_self_ty.kind() {
            let predicates = tcx.predicates_of(def_id).instantiate_own(tcx, substs);
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
        obligation: &TraitObligation<'tcx>,
        param: ty::PolyTraitRef<'tcx>,
    ) -> Vec<PredicateObligation<'tcx>> {
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
        obligation: &TraitObligation<'tcx>,
        has_nested: bool,
    ) -> ImplSourceBuiltinData<PredicateObligation<'tcx>> {
        debug!(?obligation, ?has_nested, "confirm_builtin_candidate");

        let lang_items = self.tcx().lang_items();
        let obligations = if has_nested {
            let trait_def = obligation.predicate.def_id();
            let conditions = if Some(trait_def) == lang_items.sized_trait() {
                self.sized_conditions(obligation)
            } else if Some(trait_def) == lang_items.copy_trait() {
                self.copy_clone_conditions(obligation)
            } else if Some(trait_def) == lang_items.clone_trait() {
                self.copy_clone_conditions(obligation)
            } else {
                bug!("unexpected builtin trait {:?}", trait_def)
            };
            let BuiltinImplConditions::Where(nested) = conditions else {
                bug!("obligation {:?} had matched a builtin impl but now doesn't", obligation);
            };

            let cause = obligation.derived_cause(BuiltinDerivedObligation);
            ensure_sufficient_stack(|| {
                self.collect_predicates_for_types(
                    obligation.param_env,
                    cause,
                    obligation.recursion_depth + 1,
                    trait_def,
                    nested,
                )
            })
        } else {
            vec![]
        };

        debug!(?obligations);

        ImplSourceBuiltinData { nested: obligations }
    }

    fn confirm_transmutability_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<ImplSourceBuiltinData<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        debug!(?obligation, "confirm_transmutability_candidate");

        let predicate = obligation.predicate;

        let type_at = |i| predicate.map_bound(|p| p.trait_ref.substs.type_at(i));
        let const_at = |i| predicate.skip_binder().trait_ref.substs.const_at(i);

        let src_and_dst = predicate.map_bound(|p| rustc_transmute::Types {
            dst: p.trait_ref.substs.type_at(0),
            src: p.trait_ref.substs.type_at(1),
        });

        let scope = type_at(2).skip_binder();

        let Some(assume) =
            rustc_transmute::Assume::from_const(self.infcx.tcx, obligation.param_env, const_at(3)) else {
                return Err(Unimplemented);
            };

        let cause = obligation.cause.clone();

        let mut transmute_env = rustc_transmute::TransmuteTypeEnv::new(self.infcx);

        let maybe_transmutable = transmute_env.is_transmutable(cause, src_and_dst, scope, assume);

        use rustc_transmute::Answer;

        match maybe_transmutable {
            Answer::Yes => Ok(ImplSourceBuiltinData { nested: vec![] }),
            _ => Err(Unimplemented),
        }
    }

    /// This handles the case where an `auto trait Foo` impl is being used.
    /// The idea is that the impl applies to `X : Foo` if the following conditions are met:
    ///
    /// 1. For each constituent type `Y` in `X`, `Y : Foo` holds
    /// 2. For each where-clause `C` declared on `Foo`, `[Self => X] C` holds.
    fn confirm_auto_impl_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> ImplSourceAutoImplData<PredicateObligation<'tcx>> {
        debug!(?obligation, "confirm_auto_impl_candidate");

        let self_ty = self.infcx.shallow_resolve(obligation.predicate.self_ty());
        let types = self.constituent_types_for_ty(self_ty);
        self.vtable_auto_impl(obligation, obligation.predicate.def_id(), types)
    }

    /// See `confirm_auto_impl_candidate`.
    fn vtable_auto_impl(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        trait_def_id: DefId,
        nested: ty::Binder<'tcx, Vec<Ty<'tcx>>>,
    ) -> ImplSourceAutoImplData<PredicateObligation<'tcx>> {
        debug!(?nested, "vtable_auto_impl");
        ensure_sufficient_stack(|| {
            let cause = obligation.derived_cause(BuiltinDerivedObligation);

            let poly_trait_ref = obligation.predicate.to_poly_trait_ref();
            let trait_ref = self.infcx.instantiate_binder_with_placeholders(poly_trait_ref);
            let trait_obligations: Vec<PredicateObligation<'_>> = self.impl_or_trait_obligations(
                &cause,
                obligation.recursion_depth + 1,
                obligation.param_env,
                trait_def_id,
                &trait_ref.substs,
                obligation.predicate,
            );

            let mut obligations = self.collect_predicates_for_types(
                obligation.param_env,
                cause,
                obligation.recursion_depth + 1,
                trait_def_id,
                nested,
            );

            // Adds the predicates from the trait. Note that this contains a `Self: Trait`
            // predicate as usual. It won't have any effect since auto traits are coinductive.
            obligations.extend(trait_obligations);

            debug!(?obligations, "vtable_auto_impl");

            ImplSourceAutoImplData { trait_def_id, nested: obligations }
        })
    }

    fn confirm_impl_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        impl_def_id: DefId,
    ) -> ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>> {
        debug!(?obligation, ?impl_def_id, "confirm_impl_candidate");

        // First, create the substitutions by matching the impl again,
        // this time not in a probe.
        let substs = self.rematch_impl(impl_def_id, obligation);
        debug!(?substs, "impl substs");
        ensure_sufficient_stack(|| {
            self.vtable_impl(
                impl_def_id,
                substs,
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
        substs: Normalized<'tcx, SubstsRef<'tcx>>,
        cause: &ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
        parent_trait_pred: ty::Binder<'tcx, ty::TraitPredicate<'tcx>>,
    ) -> ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>> {
        debug!(?impl_def_id, ?substs, ?recursion_depth, "vtable_impl");

        let mut impl_obligations = self.impl_or_trait_obligations(
            cause,
            recursion_depth,
            param_env,
            impl_def_id,
            &substs.value,
            parent_trait_pred,
        );

        debug!(?impl_obligations, "vtable_impl");

        // Because of RFC447, the impl-trait-ref and obligations
        // are sufficient to determine the impl substs, without
        // relying on projections in the impl-trait-ref.
        //
        // e.g., `impl<U: Tr, V: Iterator<Item=U>> Foo<<U as Tr>::T> for V`
        impl_obligations.extend(substs.obligations);

        ImplSourceUserDefinedData { impl_def_id, substs: substs.value, nested: impl_obligations }
    }

    fn confirm_object_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        index: usize,
    ) -> Result<ImplSourceObjectData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let tcx = self.tcx();
        debug!(?obligation, ?index, "confirm_object_candidate");

        let trait_predicate = self.infcx.instantiate_binder_with_placeholders(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(trait_predicate.self_ty());
        let obligation_trait_ref = ty::Binder::dummy(trait_predicate.trait_ref);
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

        let mut nested = vec![];

        let mut supertraits = util::supertraits(tcx, ty::Binder::dummy(object_trait_ref));
        let unnormalized_upcast_trait_ref =
            supertraits.nth(index).expect("supertraits iterator no longer has as many elements");

        let upcast_trait_ref = normalize_with_depth_to(
            self,
            obligation.param_env,
            obligation.cause.clone(),
            obligation.recursion_depth + 1,
            unnormalized_upcast_trait_ref,
            &mut nested,
        );

        nested.extend(self.infcx.commit_if_ok(|_| {
            self.infcx
                .at(&obligation.cause, obligation.param_env)
                .sup(obligation_trait_ref, upcast_trait_ref)
                .map(|InferOk { obligations, .. }| obligations)
                .map_err(|_| Unimplemented)
        })?);

        // Check supertraits hold. This is so that their associated type bounds
        // will be checked in the code below.
        for super_trait in tcx
            .super_predicates_of(trait_predicate.def_id())
            .instantiate(tcx, trait_predicate.trait_ref.substs)
            .predicates
            .into_iter()
        {
            let normalized_super_trait = normalize_with_depth_to(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                super_trait,
                &mut nested,
            );
            nested.push(obligation.with(tcx, normalized_super_trait));
        }

        let assoc_types: Vec<_> = tcx
            .associated_items(trait_predicate.def_id())
            .in_definition_order()
            .filter_map(
                |item| if item.kind == ty::AssocKind::Type { Some(item.def_id) } else { None },
            )
            .collect();

        for assoc_type in assoc_types {
            let defs: &ty::Generics = tcx.generics_of(assoc_type);

            if !defs.params.is_empty() && !tcx.features().generic_associated_types_extended {
                tcx.sess.delay_span_bug(
                    obligation.cause.span,
                    "GATs in trait object shouldn't have been considered",
                );
                return Err(SelectionError::Unimplemented);
            }

            // This maybe belongs in wf, but that can't (doesn't) handle
            // higher-ranked things.
            // Prevent, e.g., `dyn Iterator<Item = str>`.
            for bound in self.tcx().item_bounds(assoc_type).transpose_iter() {
                let subst_bound =
                    if defs.count() == 0 {
                        bound.subst(tcx, trait_predicate.trait_ref.substs)
                    } else {
                        let mut substs = smallvec::SmallVec::with_capacity(defs.count());
                        substs.extend(trait_predicate.trait_ref.substs.iter());
                        let mut bound_vars: smallvec::SmallVec<[ty::BoundVariableKind; 8]> =
                            smallvec::SmallVec::with_capacity(
                                bound.0.kind().bound_vars().len() + defs.count(),
                            );
                        bound_vars.extend(bound.0.kind().bound_vars().into_iter());
                        InternalSubsts::fill_single(&mut substs, defs, &mut |param, _| match param
                            .kind
                        {
                            GenericParamDefKind::Type { .. } => {
                                let kind = ty::BoundTyKind::Param(param.def_id, param.name);
                                let bound_var = ty::BoundVariableKind::Ty(kind);
                                bound_vars.push(bound_var);
                                tcx.mk_bound(
                                    ty::INNERMOST,
                                    ty::BoundTy {
                                        var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                                        kind,
                                    },
                                )
                                .into()
                            }
                            GenericParamDefKind::Lifetime => {
                                let kind = ty::BoundRegionKind::BrNamed(param.def_id, param.name);
                                let bound_var = ty::BoundVariableKind::Region(kind);
                                bound_vars.push(bound_var);
                                tcx.mk_re_late_bound(
                                    ty::INNERMOST,
                                    ty::BoundRegion {
                                        var: ty::BoundVar::from_usize(bound_vars.len() - 1),
                                        kind,
                                    },
                                )
                                .into()
                            }
                            GenericParamDefKind::Const { .. } => {
                                let bound_var = ty::BoundVariableKind::Const;
                                bound_vars.push(bound_var);
                                tcx.mk_const(
                                    ty::ConstKind::Bound(
                                        ty::INNERMOST,
                                        ty::BoundVar::from_usize(bound_vars.len() - 1),
                                    ),
                                    tcx.type_of(param.def_id)
                                        .no_bound_vars()
                                        .expect("const parameter types cannot be generic"),
                                )
                                .into()
                            }
                        });
                        let bound_vars = tcx.intern_bound_variable_kinds(&bound_vars);
                        let assoc_ty_substs = tcx.intern_substs(&substs);
                        let bound =
                            bound.map_bound(|b| b.kind().skip_binder()).subst(tcx, assoc_ty_substs);
                        tcx.mk_predicate(ty::Binder::bind_with_vars(bound, bound_vars))
                    };
                let normalized_bound = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    subst_bound,
                    &mut nested,
                );
                nested.push(obligation.with(tcx, normalized_bound));
            }
        }

        debug!(?nested, "object nested obligations");

        let vtable_base = vtable_trait_first_method_offset(
            tcx,
            (unnormalized_upcast_trait_ref, ty::Binder::dummy(object_trait_ref)),
        );

        Ok(ImplSourceObjectData { upcast_trait_ref, vtable_base, nested })
    }

    fn confirm_fn_pointer_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        is_const: bool,
    ) -> Result<ImplSourceFnPointerData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>>
    {
        debug!(?obligation, "confirm_fn_pointer_candidate");

        let tcx = self.tcx();
        let self_ty = self
            .infcx
            .shallow_resolve(obligation.self_ty().no_bound_vars())
            .expect("fn pointer should not capture bound vars from predicate");
        let sig = self_ty.fn_sig(tcx);
        let trait_ref = closure_trait_ref_and_return_type(
            tcx,
            obligation.predicate.def_id(),
            self_ty,
            sig,
            util::TupleArgumentsFlag::Yes,
        )
        .map_bound(|(trait_ref, _)| trait_ref);

        let mut nested = self.confirm_poly_trait_refs(obligation, trait_ref)?;
        let cause = obligation.derived_cause(BuiltinDerivedObligation);

        if obligation.is_const() && !is_const {
            // function is a trait method
            if let ty::FnDef(def_id, substs) = self_ty.kind() && let Some(trait_id) = tcx.trait_of_item(*def_id) {
                let trait_ref = TraitRef::from_method(tcx, trait_id, *substs);
                let poly_trait_pred = Binder::dummy(trait_ref).with_constness(ty::BoundConstness::ConstIfConst);
                let obligation = Obligation::new(tcx, cause.clone(), obligation.param_env, poly_trait_pred);
                nested.push(obligation);
            }
        }

        // Confirm the `type Output: Sized;` bound that is present on `FnOnce`
        let output_ty = self.infcx.instantiate_binder_with_placeholders(sig.output());
        let output_ty = normalize_with_depth_to(
            self,
            obligation.param_env,
            cause.clone(),
            obligation.recursion_depth,
            output_ty,
            &mut nested,
        );
        let tr =
            ty::Binder::dummy(self.tcx().at(cause.span).mk_trait_ref(LangItem::Sized, [output_ty]));
        nested.push(Obligation::new(self.infcx.tcx, cause, obligation.param_env, tr));

        Ok(ImplSourceFnPointerData { fn_ty: self_ty, nested })
    }

    fn confirm_trait_alias_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> ImplSourceTraitAliasData<'tcx, PredicateObligation<'tcx>> {
        debug!(?obligation, "confirm_trait_alias_candidate");

        let alias_def_id = obligation.predicate.def_id();
        let predicate = self.infcx.instantiate_binder_with_placeholders(obligation.predicate);
        let trait_ref = predicate.trait_ref;
        let trait_def_id = trait_ref.def_id;
        let substs = trait_ref.substs;

        let trait_obligations = self.impl_or_trait_obligations(
            &obligation.cause,
            obligation.recursion_depth,
            obligation.param_env,
            trait_def_id,
            &substs,
            obligation.predicate,
        );

        debug!(?trait_def_id, ?trait_obligations, "trait alias obligations");

        ImplSourceTraitAliasData { alias_def_id, substs, nested: trait_obligations }
    }

    fn confirm_generator_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<ImplSourceGeneratorData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>>
    {
        // Okay to skip binder because the substs on generator types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        let ty::Generator(generator_def_id, substs, _) = *self_ty.kind() else {
            bug!("closure candidate for non-closure {:?}", obligation);
        };

        debug!(?obligation, ?generator_def_id, ?substs, "confirm_generator_candidate");

        let gen_sig = substs.as_generator().poly_sig();

        // NOTE: The self-type is a generator type and hence is
        // in fact unparameterized (or at least does not reference any
        // regions bound in the obligation).
        let self_ty = obligation
            .predicate
            .self_ty()
            .no_bound_vars()
            .expect("unboxed closure type should not capture bound vars from the predicate");

        let trait_ref = super::util::generator_trait_ref_and_outputs(
            self.tcx(),
            obligation.predicate.def_id(),
            self_ty,
            gen_sig,
        )
        .map_bound(|(trait_ref, ..)| trait_ref);

        let nested = self.confirm_poly_trait_refs(obligation, trait_ref)?;
        debug!(?trait_ref, ?nested, "generator candidate obligations");

        Ok(ImplSourceGeneratorData { generator_def_id, substs, nested })
    }

    fn confirm_future_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<ImplSourceFutureData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        // Okay to skip binder because the substs on generator types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        let ty::Generator(generator_def_id, substs, _) = *self_ty.kind() else {
            bug!("closure candidate for non-closure {:?}", obligation);
        };

        debug!(?obligation, ?generator_def_id, ?substs, "confirm_future_candidate");

        let gen_sig = substs.as_generator().poly_sig();

        let trait_ref = super::util::future_trait_ref_and_outputs(
            self.tcx(),
            obligation.predicate.def_id(),
            obligation.predicate.no_bound_vars().expect("future has no bound vars").self_ty(),
            gen_sig,
        )
        .map_bound(|(trait_ref, ..)| trait_ref);

        let nested = self.confirm_poly_trait_refs(obligation, trait_ref)?;
        debug!(?trait_ref, ?nested, "future candidate obligations");

        Ok(ImplSourceFutureData { generator_def_id, substs, nested })
    }

    #[instrument(skip(self), level = "debug")]
    fn confirm_closure_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<ImplSourceClosureData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let kind = self
            .tcx()
            .fn_trait_kind_from_def_id(obligation.predicate.def_id())
            .unwrap_or_else(|| bug!("closure candidate for non-fn trait {:?}", obligation));

        // Okay to skip binder because the substs on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        let ty::Closure(closure_def_id, substs) = *self_ty.kind() else {
            bug!("closure candidate for non-closure {:?}", obligation);
        };

        let trait_ref = self.closure_trait_ref_unnormalized(obligation, substs);
        let mut nested = self.confirm_poly_trait_refs(obligation, trait_ref)?;

        debug!(?closure_def_id, ?trait_ref, ?nested, "confirm closure candidate obligations");

        // FIXME: Chalk
        if self.tcx().sess.opts.unstable_opts.trait_solver != TraitSolver::Chalk {
            nested.push(obligation.with(
                self.tcx(),
                ty::Binder::dummy(ty::PredicateKind::ClosureKind(closure_def_id, substs, kind)),
            ));
        }

        Ok(ImplSourceClosureData { closure_def_id, substs, nested })
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
    fn confirm_poly_trait_refs(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        expected_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<Vec<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let obligation_trait_ref = obligation.predicate.to_poly_trait_ref();
        // Normalize the obligation and expected trait refs together, because why not
        let Normalized { obligations: nested, value: (obligation_trait_ref, expected_trait_ref) } =
            ensure_sufficient_stack(|| {
                normalize_with_depth(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    (obligation_trait_ref, expected_trait_ref),
                )
            });

        self.infcx
            .at(&obligation.cause, obligation.param_env)
            .sup(obligation_trait_ref, expected_trait_ref)
            .map(|InferOk { mut obligations, .. }| {
                obligations.extend(nested);
                obligations
            })
            .map_err(|e| OutputTypeParameterMismatch(expected_trait_ref, obligation_trait_ref, e))
    }

    fn confirm_trait_upcasting_unsize_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        idx: usize,
    ) -> Result<ImplSourceTraitUpcastingData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>>
    {
        let tcx = self.tcx();

        // `assemble_candidates_for_unsizing` should ensure there are no late-bound
        // regions here. See the comment there for more details.
        let source = self.infcx.shallow_resolve(obligation.self_ty().no_bound_vars().unwrap());
        let target = obligation.predicate.skip_binder().trait_ref.substs.type_at(1);
        let target = self.infcx.shallow_resolve(target);

        debug!(?source, ?target, "confirm_trait_upcasting_unsize_candidate");

        let mut nested = vec![];
        let source_trait_ref;
        let upcast_trait_ref;
        match (source.kind(), target.kind()) {
            // TraitA+Kx+'a -> TraitB+Ky+'b (trait upcasting coercion).
            (
                &ty::Dynamic(ref data_a, r_a, repr_a @ ty::Dyn),
                &ty::Dynamic(ref data_b, r_b, ty::Dyn),
            ) => {
                // See `assemble_candidates_for_unsizing` for more info.
                // We already checked the compatibility of auto traits within `assemble_candidates_for_unsizing`.
                let principal_a = data_a.principal().unwrap();
                source_trait_ref = principal_a.with_self_ty(tcx, source);
                upcast_trait_ref = util::supertraits(tcx, source_trait_ref).nth(idx).unwrap();
                assert_eq!(data_b.principal_def_id(), Some(upcast_trait_ref.def_id()));
                let existential_predicate = upcast_trait_ref.map_bound(|trait_ref| {
                    ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::erase_self_ty(
                        tcx, trait_ref,
                    ))
                });
                let iter = Some(existential_predicate)
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
                    );
                let existential_predicates = tcx.mk_poly_existential_predicates(iter);
                let source_trait = tcx.mk_dynamic(existential_predicates, r_b, repr_a);

                // Require that the traits involved in this upcast are **equal**;
                // only the **lifetime bound** is changed.
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .sup(target, source_trait)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Register one obligation for 'a: 'b.
                let cause = ObligationCause::new(
                    obligation.cause.span,
                    obligation.cause.body_id,
                    ObjectCastObligation(source, target),
                );
                let outlives = ty::OutlivesPredicate(r_a, r_b);
                nested.push(Obligation::with_depth(
                    tcx,
                    cause,
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    obligation.predicate.rebind(outlives),
                ));
            }
            _ => bug!(),
        };

        let vtable_segment_callback = {
            let mut vptr_offset = 0;
            move |segment| {
                match segment {
                    VtblSegment::MetadataDSA => {
                        vptr_offset += TyCtxt::COMMON_VTABLE_ENTRIES.len();
                    }
                    VtblSegment::TraitOwnEntries { trait_ref, emit_vptr } => {
                        vptr_offset += count_own_vtable_entries(tcx, trait_ref);
                        if trait_ref == upcast_trait_ref {
                            if emit_vptr {
                                return ControlFlow::Break(Some(vptr_offset));
                            } else {
                                return ControlFlow::Break(None);
                            }
                        }

                        if emit_vptr {
                            vptr_offset += 1;
                        }
                    }
                }
                ControlFlow::Continue(())
            }
        };

        let vtable_vptr_slot =
            prepare_vtable_segments(tcx, source_trait_ref, vtable_segment_callback).unwrap();

        Ok(ImplSourceTraitUpcastingData { upcast_trait_ref, vtable_vptr_slot, nested })
    }

    fn confirm_builtin_unsize_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<ImplSourceBuiltinData<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        let tcx = self.tcx();

        // `assemble_candidates_for_unsizing` should ensure there are no late-bound
        // regions here. See the comment there for more details.
        let source = self.infcx.shallow_resolve(obligation.self_ty().no_bound_vars().unwrap());
        let target = obligation.predicate.skip_binder().trait_ref.substs.type_at(1);
        let target = self.infcx.shallow_resolve(target);

        debug!(?source, ?target, "confirm_builtin_unsize_candidate");

        let mut nested = vec![];
        match (source.kind(), target.kind()) {
            // Trait+Kx+'a -> Trait+Ky+'b (auto traits and lifetime subtyping).
            (&ty::Dynamic(ref data_a, r_a, dyn_a), &ty::Dynamic(ref data_b, r_b, dyn_b))
                if dyn_a == dyn_b =>
            {
                // See `assemble_candidates_for_unsizing` for more info.
                // We already checked the compatibility of auto traits within `assemble_candidates_for_unsizing`.
                let iter = data_a
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
                    );
                let existential_predicates = tcx.mk_poly_existential_predicates(iter);
                let source_trait = tcx.mk_dynamic(existential_predicates, r_b, dyn_a);

                // Require that the traits involved in this upcast are **equal**;
                // only the **lifetime bound** is changed.
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .sup(target, source_trait)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Register one obligation for 'a: 'b.
                let cause = ObligationCause::new(
                    obligation.cause.span,
                    obligation.cause.body_id,
                    ObjectCastObligation(source, target),
                );
                let outlives = ty::OutlivesPredicate(r_a, r_b);
                nested.push(Obligation::with_depth(
                    tcx,
                    cause,
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    obligation.predicate.rebind(outlives),
                ));
            }

            // `T` -> `Trait`
            (_, &ty::Dynamic(ref data, r, ty::Dyn)) => {
                let mut object_dids = data.auto_traits().chain(data.principal_def_id());
                if let Some(did) = object_dids.find(|did| !tcx.check_is_object_safe(*did)) {
                    return Err(TraitNotObjectSafe(did));
                }

                let cause = ObligationCause::new(
                    obligation.cause.span,
                    obligation.cause.body_id,
                    ObjectCastObligation(source, target),
                );

                let predicate_to_obligation = |predicate| {
                    Obligation::with_depth(
                        tcx,
                        cause.clone(),
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
                nested.extend(
                    data.iter().map(|predicate| {
                        predicate_to_obligation(predicate.with_self_ty(tcx, source))
                    }),
                );

                // We can only make objects from sized types.
                let tr =
                    ty::Binder::dummy(tcx.at(cause.span).mk_trait_ref(LangItem::Sized, [source]));
                nested.push(predicate_to_obligation(tr.without_const().to_predicate(tcx)));

                // If the type is `Foo + 'a`, ensure that the type
                // being cast to `Foo + 'a` outlives `'a`:
                let outlives = ty::OutlivesPredicate(source, r);
                nested.push(predicate_to_obligation(ty::Binder::dummy(outlives).to_predicate(tcx)));
            }

            // `[T; n]` -> `[T]`
            (&ty::Array(a, _), &ty::Slice(b)) => {
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(b, a)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);
            }

            // `Struct<T>` -> `Struct<U>`
            (&ty::Adt(def, substs_a), &ty::Adt(_, substs_b)) => {
                let unsizing_params = tcx.unsizing_params_for_adt(def.did());
                if unsizing_params.is_empty() {
                    return Err(Unimplemented);
                }

                let tail_field = def
                    .non_enum_variant()
                    .fields
                    .last()
                    .expect("expected unsized ADT to have a tail field");
                let tail_field_ty = tcx.type_of(tail_field.did);

                // Extract `TailField<T>` and `TailField<U>` from `Struct<T>` and `Struct<U>`,
                // normalizing in the process, since `type_of` returns something directly from
                // astconv (which means it's un-normalized).
                let source_tail = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    tail_field_ty.subst(tcx, substs_a),
                    &mut nested,
                );
                let target_tail = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    tail_field_ty.subst(tcx, substs_b),
                    &mut nested,
                );

                // Check that the source struct with the target's
                // unsizing parameters is equal to the target.
                let substs = tcx.mk_substs(substs_a.iter().enumerate().map(|(i, k)| {
                    if unsizing_params.contains(i as u32) { substs_b[i] } else { k }
                }));
                let new_struct = tcx.mk_adt(def, substs);
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(target, new_struct)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Construct the nested `TailField<T>: Unsize<TailField<U>>` predicate.
                nested.push(predicate_for_trait_def(
                    tcx,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.predicate.def_id(),
                    obligation.recursion_depth + 1,
                    [source_tail, target_tail],
                ));
            }

            // `(.., T)` -> `(.., U)`
            (&ty::Tuple(tys_a), &ty::Tuple(tys_b)) => {
                assert_eq!(tys_a.len(), tys_b.len());

                // The last field of the tuple has to exist.
                let (&a_last, a_mid) = tys_a.split_last().ok_or(Unimplemented)?;
                let &b_last = tys_b.last().unwrap();

                // Check that the source tuple with the target's
                // last element is equal to the target.
                let new_tuple = tcx.mk_tup(a_mid.iter().copied().chain(iter::once(b_last)));
                let InferOk { obligations, .. } = self
                    .infcx
                    .at(&obligation.cause, obligation.param_env)
                    .eq(target, new_tuple)
                    .map_err(|_| Unimplemented)?;
                nested.extend(obligations);

                // Construct the nested `T: Unsize<U>` predicate.
                nested.push(ensure_sufficient_stack(|| {
                    predicate_for_trait_def(
                        tcx,
                        obligation.param_env,
                        obligation.cause.clone(),
                        obligation.predicate.def_id(),
                        obligation.recursion_depth + 1,
                        [a_last, b_last],
                    )
                }));
            }

            _ => bug!("source: {source}, target: {target}"),
        };

        Ok(ImplSourceBuiltinData { nested })
    }

    fn confirm_const_destruct_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        impl_def_id: Option<DefId>,
    ) -> Result<ImplSourceConstDestructData<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        // `~const Destruct` in a non-const environment is always trivially true, since our type is `Drop`
        if !obligation.is_const() {
            return Ok(ImplSourceConstDestructData { nested: vec![] });
        }

        let drop_trait = self.tcx().require_lang_item(LangItem::Drop, None);

        let tcx = self.tcx();
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty());

        let mut nested = vec![];
        let cause = obligation.derived_cause(BuiltinDerivedObligation);

        // If we have a custom `impl const Drop`, then
        // first check it like a regular impl candidate.
        // This is copied from confirm_impl_candidate but remaps the predicate to `~const Drop` beforehand.
        if let Some(impl_def_id) = impl_def_id {
            let mut new_obligation = obligation.clone();
            new_obligation.predicate = new_obligation.predicate.map_bound(|mut trait_pred| {
                trait_pred.trait_ref.def_id = drop_trait;
                trait_pred
            });
            let substs = self.rematch_impl(impl_def_id, &new_obligation);
            debug!(?substs, "impl substs");

            let cause = obligation.derived_cause(|derived| {
                ImplDerivedObligation(Box::new(ImplDerivedObligationCause {
                    derived,
                    impl_or_alias_def_id: impl_def_id,
                    impl_def_predicate_index: None,
                    span: obligation.cause.span,
                }))
            });
            let obligations = ensure_sufficient_stack(|| {
                self.vtable_impl(
                    impl_def_id,
                    substs,
                    &cause,
                    new_obligation.recursion_depth + 1,
                    new_obligation.param_env,
                    obligation.predicate,
                )
            });
            nested.extend(obligations.nested);
        }

        // We want to confirm the ADT's fields if we have an ADT
        let mut stack = match *self_ty.skip_binder().kind() {
            ty::Adt(def, substs) => def.all_fields().map(|f| f.ty(tcx, substs)).collect(),
            _ => vec![self_ty.skip_binder()],
        };

        while let Some(nested_ty) = stack.pop() {
            match *nested_ty.kind() {
                // We know these types are trivially drop
                ty::Bool
                | ty::Char
                | ty::Int(_)
                | ty::Uint(_)
                | ty::Float(_)
                | ty::Infer(ty::IntVar(_))
                | ty::Infer(ty::FloatVar(_))
                | ty::Str
                | ty::RawPtr(_)
                | ty::Ref(..)
                | ty::FnDef(..)
                | ty::FnPtr(_)
                | ty::Never
                | ty::Foreign(_) => {}

                // `ManuallyDrop` is trivially drop
                ty::Adt(def, _) if Some(def.did()) == tcx.lang_items().manually_drop() => {}

                // These types are built-in, so we can fast-track by registering
                // nested predicates for their constituent type(s)
                ty::Array(ty, _) | ty::Slice(ty) => {
                    stack.push(ty);
                }
                ty::Tuple(tys) => {
                    stack.extend(tys.iter());
                }
                ty::Closure(_, substs) => {
                    stack.push(substs.as_closure().tupled_upvars_ty());
                }
                ty::Generator(_, substs, _) => {
                    let generator = substs.as_generator();
                    stack.extend([generator.tupled_upvars_ty(), generator.witness()]);
                }
                ty::GeneratorWitness(tys) => {
                    stack.extend(tcx.erase_late_bound_regions(tys).to_vec());
                }
                ty::GeneratorWitnessMIR(def_id, substs) => {
                    let tcx = self.tcx();
                    stack.extend(tcx.generator_hidden_types(def_id).map(|bty| {
                        let ty = bty.subst(tcx, substs);
                        debug_assert!(!ty.has_late_bound_regions());
                        ty
                    }))
                }

                // If we have a projection type, make sure to normalize it so we replace it
                // with a fresh infer variable
                ty::Alias(ty::Projection, ..) => {
                    let predicate = normalize_with_depth_to(
                        self,
                        obligation.param_env,
                        cause.clone(),
                        obligation.recursion_depth + 1,
                        self_ty.rebind(ty::TraitPredicate {
                            trait_ref: self
                                .tcx()
                                .at(cause.span)
                                .mk_trait_ref(LangItem::Destruct, [nested_ty]),
                            constness: ty::BoundConstness::ConstIfConst,
                            polarity: ty::ImplPolarity::Positive,
                        }),
                        &mut nested,
                    );

                    nested.push(Obligation::with_depth(
                        tcx,
                        cause.clone(),
                        obligation.recursion_depth + 1,
                        obligation.param_env,
                        predicate,
                    ));
                }

                // If we have any other type (e.g. an ADT), just register a nested obligation
                // since it's either not `const Drop` (and we raise an error during selection),
                // or it's an ADT (and we need to check for a custom impl during selection)
                _ => {
                    let predicate = self_ty.rebind(ty::TraitPredicate {
                        trait_ref: self
                            .tcx()
                            .at(cause.span)
                            .mk_trait_ref(LangItem::Destruct, [nested_ty]),
                        constness: ty::BoundConstness::ConstIfConst,
                        polarity: ty::ImplPolarity::Positive,
                    });

                    nested.push(Obligation::with_depth(
                        tcx,
                        cause.clone(),
                        obligation.recursion_depth + 1,
                        obligation.param_env,
                        predicate,
                    ));
                }
            }
        }

        Ok(ImplSourceConstDestructData { nested })
    }
}
