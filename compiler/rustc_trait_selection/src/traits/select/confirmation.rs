//! Confirmation.
//!
//! Confirmation unifies the output type parameters of the trait
//! with the values found in the obligation, possibly yielding a
//! type error.  See the [rustc dev guide] for more details.
//!
//! [rustc dev guide]:
//! https://rustc-dev-guide.rust-lang.org/traits/resolution.html#confirmation
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::lang_items::LangItem;
use rustc_hir::Constness;
use rustc_index::bit_set::GrowableBitSet;
use rustc_infer::infer::InferOk;
use rustc_infer::infer::LateBoundRegionConversionTime::HigherRankedType;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, Subst, SubstsRef};
use rustc_middle::ty::{self, Ty};
use rustc_middle::ty::{ToPolyTraitRef, ToPredicate, WithConstness};
use rustc_span::def_id::DefId;

use crate::traits::project::{normalize_with_depth, normalize_with_depth_to};
use crate::traits::select::TraitObligationExt;
use crate::traits::util;
use crate::traits::util::{closure_trait_ref_and_return_type, predicate_for_trait_def};
use crate::traits::ImplSource;
use crate::traits::Normalized;
use crate::traits::OutputTypeParameterMismatch;
use crate::traits::Selection;
use crate::traits::TraitNotObjectSafe;
use crate::traits::{BuiltinDerivedObligation, ImplDerivedObligation};
use crate::traits::{
    ImplSourceAutoImplData, ImplSourceBuiltinData, ImplSourceClosureData,
    ImplSourceDiscriminantKindData, ImplSourceFnPointerData, ImplSourceGeneratorData,
    ImplSourceObjectData, ImplSourceTraitAliasData, ImplSourceUserDefinedData,
};
use crate::traits::{ObjectCastObligation, PredicateObligation, TraitObligation};
use crate::traits::{Obligation, ObligationCause};
use crate::traits::{SelectionError, Unimplemented};

use super::BuiltinImplConditions;
use super::SelectionCandidate::{self, *};
use super::SelectionContext;

use std::iter;

impl<'cx, 'tcx> SelectionContext<'cx, 'tcx> {
    #[instrument(level = "debug", skip(self))]
    pub(super) fn confirm_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        candidate: SelectionCandidate<'tcx>,
    ) -> Result<Selection<'tcx>, SelectionError<'tcx>> {
        match candidate {
            BuiltinCandidate { has_nested } => {
                let data = self.confirm_builtin_candidate(obligation, has_nested);
                Ok(ImplSource::Builtin(data))
            }

            ParamCandidate(param) => {
                let obligations = self.confirm_param_candidate(obligation, param.value);
                Ok(ImplSource::Param(obligations, param.constness))
            }

            ImplCandidate(impl_def_id) => {
                Ok(ImplSource::UserDefined(self.confirm_impl_candidate(obligation, impl_def_id)))
            }

            AutoImplCandidate(trait_def_id) => {
                let data = self.confirm_auto_impl_candidate(obligation, trait_def_id);
                Ok(ImplSource::AutoImpl(data))
            }

            ProjectionCandidate(idx) => {
                let obligations = self.confirm_projection_candidate(obligation, idx)?;
                // FIXME(jschievink): constness
                Ok(ImplSource::Param(obligations, Constness::NotConst))
            }

            ObjectCandidate(idx) => {
                let data = self.confirm_object_candidate(obligation, idx)?;
                Ok(ImplSource::Object(data))
            }

            ClosureCandidate => {
                let vtable_closure = self.confirm_closure_candidate(obligation)?;
                Ok(ImplSource::Closure(vtable_closure))
            }

            GeneratorCandidate => {
                let vtable_generator = self.confirm_generator_candidate(obligation)?;
                Ok(ImplSource::Generator(vtable_generator))
            }

            FnPointerCandidate => {
                let data = self.confirm_fn_pointer_candidate(obligation)?;
                Ok(ImplSource::FnPointer(data))
            }

            DiscriminantKindCandidate => {
                Ok(ImplSource::DiscriminantKind(ImplSourceDiscriminantKindData))
            }

            TraitAliasCandidate(alias_def_id) => {
                let data = self.confirm_trait_alias_candidate(obligation, alias_def_id);
                Ok(ImplSource::TraitAlias(data))
            }

            BuiltinObjectCandidate => {
                // This indicates something like `Trait + Send: Send`. In this case, we know that
                // this holds because that's what the object type is telling us, and there's really
                // no additional obligations to prove and no types in particular to unify, etc.
                Ok(ImplSource::Param(Vec::new(), Constness::NotConst))
            }

            BuiltinUnsizeCandidate => {
                let data = self.confirm_builtin_unsize_candidate(obligation)?;
                Ok(ImplSource::Builtin(data))
            }
        }
    }

    fn confirm_projection_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        idx: usize,
    ) -> Result<Vec<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        self.infcx.commit_unconditionally(|_| {
            let tcx = self.tcx();

            let trait_predicate = self.infcx.shallow_resolve(obligation.predicate);
            let placeholder_trait_predicate =
                self.infcx().replace_bound_vars_with_placeholders(trait_predicate);
            let placeholder_self_ty = placeholder_trait_predicate.self_ty();
            let (def_id, substs) = match *placeholder_self_ty.kind() {
                ty::Projection(proj) => (proj.item_def_id, proj.substs),
                ty::Opaque(def_id, substs) => (def_id, substs),
                _ => bug!("projection candidate for unexpected type: {:?}", placeholder_self_ty),
            };

            let candidate_predicate = tcx.item_bounds(def_id)[idx].subst(tcx, substs);
            let candidate = candidate_predicate
                .to_opt_poly_trait_ref()
                .expect("projection candidate is not a trait predicate");
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
                    .sup(placeholder_trait_predicate.trait_ref.to_poly_trait_ref(), candidate.value)
                    .map(|InferOk { obligations, .. }| obligations)
                    .map_err(|_| Unimplemented)
            })?);

            if let ty::Projection(..) = placeholder_self_ty.kind() {
                for predicate in tcx.predicates_of(def_id).instantiate_own(tcx, substs).predicates {
                    let normalized = normalize_with_depth_to(
                        self,
                        obligation.param_env,
                        obligation.cause.clone(),
                        obligation.recursion_depth + 1,
                        predicate,
                        &mut obligations,
                    );
                    obligations.push(Obligation::with_depth(
                        obligation.cause.clone(),
                        obligation.recursion_depth + 1,
                        obligation.param_env,
                        normalized,
                    ));
                }
            }

            Ok(obligations)
        })
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
            let nested = match conditions {
                BuiltinImplConditions::Where(nested) => nested,
                _ => bug!("obligation {:?} had matched a builtin impl but now doesn't", obligation),
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

    /// This handles the case where a `auto trait Foo` impl is being used.
    /// The idea is that the impl applies to `X : Foo` if the following conditions are met:
    ///
    /// 1. For each constituent type `Y` in `X`, `Y : Foo` holds
    /// 2. For each where-clause `C` declared on `Foo`, `[Self => X] C` holds.
    fn confirm_auto_impl_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        trait_def_id: DefId,
    ) -> ImplSourceAutoImplData<PredicateObligation<'tcx>> {
        debug!(?obligation, ?trait_def_id, "confirm_auto_impl_candidate");

        let self_ty = self.infcx.shallow_resolve(obligation.predicate.self_ty());
        let types = self.constituent_types_for_ty(self_ty);
        self.vtable_auto_impl(obligation, trait_def_id, types)
    }

    /// See `confirm_auto_impl_candidate`.
    fn vtable_auto_impl(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        trait_def_id: DefId,
        nested: ty::Binder<Vec<Ty<'tcx>>>,
    ) -> ImplSourceAutoImplData<PredicateObligation<'tcx>> {
        debug!(?nested, "vtable_auto_impl");
        ensure_sufficient_stack(|| {
            let cause = obligation.derived_cause(BuiltinDerivedObligation);
            let mut obligations = self.collect_predicates_for_types(
                obligation.param_env,
                cause,
                obligation.recursion_depth + 1,
                trait_def_id,
                nested,
            );

            let trait_obligations: Vec<PredicateObligation<'_>> =
                self.infcx.commit_unconditionally(|_| {
                    let poly_trait_ref = obligation.predicate.to_poly_trait_ref();
                    let trait_ref = self.infcx.replace_bound_vars_with_placeholders(poly_trait_ref);
                    let cause = obligation.derived_cause(ImplDerivedObligation);
                    self.impl_or_trait_obligations(
                        cause,
                        obligation.recursion_depth + 1,
                        obligation.param_env,
                        trait_def_id,
                        &trait_ref.substs,
                    )
                });

            // Adds the predicates from the trait.  Note that this contains a `Self: Trait`
            // predicate as usual.  It won't have any effect since auto traits are coinductive.
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
        self.infcx.commit_unconditionally(|_| {
            let substs = self.rematch_impl(impl_def_id, obligation);
            debug!(?substs, "impl substs");
            let cause = obligation.derived_cause(ImplDerivedObligation);
            ensure_sufficient_stack(|| {
                self.vtable_impl(
                    impl_def_id,
                    substs,
                    cause,
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                )
            })
        })
    }

    fn vtable_impl(
        &mut self,
        impl_def_id: DefId,
        substs: Normalized<'tcx, SubstsRef<'tcx>>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
    ) -> ImplSourceUserDefinedData<'tcx, PredicateObligation<'tcx>> {
        debug!(?impl_def_id, ?substs, ?recursion_depth, "vtable_impl");

        let mut impl_obligations = self.impl_or_trait_obligations(
            cause,
            recursion_depth,
            param_env,
            impl_def_id,
            &substs.value,
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

        let trait_predicate = self.infcx.replace_bound_vars_with_placeholders(obligation.predicate);
        let self_ty = self.infcx.shallow_resolve(trait_predicate.self_ty());
        let obligation_trait_ref = ty::Binder::dummy(trait_predicate.trait_ref);
        let data = match *self_ty.kind() {
            ty::Dynamic(data, ..) => data,
            _ => span_bug!(obligation.cause.span, "object candidate with non-object"),
        };

        let object_trait_ref = data.principal().unwrap_or_else(|| {
            span_bug!(obligation.cause.span, "object candidate with no principal")
        });
        let object_trait_ref = self
            .infcx
            .replace_bound_vars_with_fresh_vars(
                obligation.cause.span,
                HigherRankedType,
                object_trait_ref,
            )
            .0;
        let object_trait_ref = object_trait_ref.with_self_ty(self.tcx(), self_ty);

        let mut nested = vec![];

        let mut supertraits = util::supertraits(tcx, ty::Binder::dummy(object_trait_ref));

        // For each of the non-matching predicates that
        // we pass over, we sum up the set of number of vtable
        // entries, so that we can compute the offset for the selected
        // trait.
        let vtable_base = supertraits
            .by_ref()
            .take(index)
            .map(|t| super::util::count_own_vtable_entries(tcx, t))
            .sum();

        let unnormalized_upcast_trait_ref =
            supertraits.next().expect("supertraits iterator no longer has as many elements");

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
            if let ty::PredicateAtom::Trait(..) = super_trait.skip_binders() {
                let normalized_super_trait = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    super_trait,
                    &mut nested,
                );
                nested.push(Obligation::new(
                    obligation.cause.clone(),
                    obligation.param_env,
                    normalized_super_trait,
                ));
            }
        }

        let assoc_types: Vec<_> = tcx
            .associated_items(trait_predicate.def_id())
            .in_definition_order()
            .filter_map(
                |item| if item.kind == ty::AssocKind::Type { Some(item.def_id) } else { None },
            )
            .collect();

        for assoc_type in assoc_types {
            if !tcx.generics_of(assoc_type).params.is_empty() {
                // FIXME(generic_associated_types) generate placeholders to
                // extend the trait substs.
                tcx.sess.span_fatal(
                    obligation.cause.span,
                    "generic associated types in trait objects are not supported yet",
                );
            }
            // This maybe belongs in wf, but that can't (doesn't) handle
            // higher-ranked things.
            // Prevent, e.g., `dyn Iterator<Item = str>`.
            for bound in self.tcx().item_bounds(assoc_type) {
                let subst_bound = bound.subst(tcx, trait_predicate.trait_ref.substs);
                let normalized_bound = normalize_with_depth_to(
                    self,
                    obligation.param_env,
                    obligation.cause.clone(),
                    obligation.recursion_depth + 1,
                    subst_bound,
                    &mut nested,
                );
                nested.push(Obligation::new(
                    obligation.cause.clone(),
                    obligation.param_env,
                    normalized_bound,
                ));
            }
        }

        debug!(?nested, "object nested obligations");
        Ok(ImplSourceObjectData { upcast_trait_ref, vtable_base, nested })
    }

    fn confirm_fn_pointer_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<ImplSourceFnPointerData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>>
    {
        debug!(?obligation, "confirm_fn_pointer_candidate");

        // Okay to skip binder; it is reintroduced below.
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        let sig = self_ty.fn_sig(self.tcx());
        let trait_ref = closure_trait_ref_and_return_type(
            self.tcx(),
            obligation.predicate.def_id(),
            self_ty,
            sig,
            util::TupleArgumentsFlag::Yes,
        )
        .map_bound(|(trait_ref, _)| trait_ref);

        let Normalized { value: trait_ref, mut obligations } = ensure_sufficient_stack(|| {
            normalize_with_depth(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                trait_ref,
            )
        });

        obligations.extend(self.confirm_poly_trait_refs(
            obligation.cause.clone(),
            obligation.param_env,
            obligation.predicate.to_poly_trait_ref(),
            trait_ref,
        )?);
        Ok(ImplSourceFnPointerData { fn_ty: self_ty, nested: obligations })
    }

    fn confirm_trait_alias_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
        alias_def_id: DefId,
    ) -> ImplSourceTraitAliasData<'tcx, PredicateObligation<'tcx>> {
        debug!(?obligation, ?alias_def_id, "confirm_trait_alias_candidate");

        self.infcx.commit_unconditionally(|_| {
            let predicate = self.infcx().replace_bound_vars_with_placeholders(obligation.predicate);
            let trait_ref = predicate.trait_ref;
            let trait_def_id = trait_ref.def_id;
            let substs = trait_ref.substs;

            let trait_obligations = self.impl_or_trait_obligations(
                obligation.cause.clone(),
                obligation.recursion_depth,
                obligation.param_env,
                trait_def_id,
                &substs,
            );

            debug!(?trait_def_id, ?trait_obligations, "trait alias obligations");

            ImplSourceTraitAliasData { alias_def_id, substs, nested: trait_obligations }
        })
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
        let (generator_def_id, substs) = match *self_ty.kind() {
            ty::Generator(id, substs, _) => (id, substs),
            _ => bug!("closure candidate for non-closure {:?}", obligation),
        };

        debug!(?obligation, ?generator_def_id, ?substs, "confirm_generator_candidate");

        let trait_ref = self.generator_trait_ref_unnormalized(obligation, substs);
        let Normalized { value: trait_ref, mut obligations } = ensure_sufficient_stack(|| {
            normalize_with_depth(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                trait_ref,
            )
        });

        debug!(?trait_ref, ?obligations, "generator candidate obligations");

        obligations.extend(self.confirm_poly_trait_refs(
            obligation.cause.clone(),
            obligation.param_env,
            obligation.predicate.to_poly_trait_ref(),
            trait_ref,
        )?);

        Ok(ImplSourceGeneratorData { generator_def_id, substs, nested: obligations })
    }

    fn confirm_closure_candidate(
        &mut self,
        obligation: &TraitObligation<'tcx>,
    ) -> Result<ImplSourceClosureData<'tcx, PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        debug!(?obligation, "confirm_closure_candidate");

        let kind = self
            .tcx()
            .fn_trait_kind_from_lang_item(obligation.predicate.def_id())
            .unwrap_or_else(|| bug!("closure candidate for non-fn trait {:?}", obligation));

        // Okay to skip binder because the substs on closure types never
        // touch bound regions, they just capture the in-scope
        // type/region parameters.
        let self_ty = self.infcx.shallow_resolve(obligation.self_ty().skip_binder());
        let (closure_def_id, substs) = match *self_ty.kind() {
            ty::Closure(id, substs) => (id, substs),
            _ => bug!("closure candidate for non-closure {:?}", obligation),
        };

        let trait_ref = self.closure_trait_ref_unnormalized(obligation, substs);
        let Normalized { value: trait_ref, mut obligations } = ensure_sufficient_stack(|| {
            normalize_with_depth(
                self,
                obligation.param_env,
                obligation.cause.clone(),
                obligation.recursion_depth + 1,
                trait_ref,
            )
        });

        debug!(?closure_def_id, ?trait_ref, ?obligations, "confirm closure candidate obligations");

        obligations.extend(self.confirm_poly_trait_refs(
            obligation.cause.clone(),
            obligation.param_env,
            obligation.predicate.to_poly_trait_ref(),
            trait_ref,
        )?);

        // FIXME: Chalk

        if !self.tcx().sess.opts.debugging_opts.chalk {
            obligations.push(Obligation::new(
                obligation.cause.clone(),
                obligation.param_env,
                ty::PredicateAtom::ClosureKind(closure_def_id, substs, kind)
                    .to_predicate(self.tcx()),
            ));
        }

        Ok(ImplSourceClosureData { closure_def_id, substs, nested: obligations })
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
    ///
    ///     impl Fn(i32) for Closure { ... }
    ///
    /// Now imagine our obligation is `Closure: Fn(usize)`. So far
    /// we have matched the self type `Closure`. At this point we'll
    /// compare the `i32` to `usize` and generate an error.
    ///
    /// Note that this checking occurs *after* the impl has selected,
    /// because these output type parameters should not affect the
    /// selection of the impl. Therefore, if there is a mismatch, we
    /// report an error to the user.
    fn confirm_poly_trait_refs(
        &mut self,
        obligation_cause: ObligationCause<'tcx>,
        obligation_param_env: ty::ParamEnv<'tcx>,
        obligation_trait_ref: ty::PolyTraitRef<'tcx>,
        expected_trait_ref: ty::PolyTraitRef<'tcx>,
    ) -> Result<Vec<PredicateObligation<'tcx>>, SelectionError<'tcx>> {
        self.infcx
            .at(&obligation_cause, obligation_param_env)
            .sup(obligation_trait_ref, expected_trait_ref)
            .map(|InferOk { obligations, .. }| obligations)
            .map_err(|e| OutputTypeParameterMismatch(expected_trait_ref, obligation_trait_ref, e))
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
            // Trait+Kx+'a -> Trait+Ky+'b (upcasts).
            (&ty::Dynamic(ref data_a, r_a), &ty::Dynamic(ref data_b, r_b)) => {
                // See `assemble_candidates_for_unsizing` for more info.
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
                let source_trait = tcx.mk_dynamic(existential_predicates, r_b);

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
                    ObjectCastObligation(target),
                );
                let outlives = ty::OutlivesPredicate(r_a, r_b);
                nested.push(Obligation::with_depth(
                    cause,
                    obligation.recursion_depth + 1,
                    obligation.param_env,
                    ty::Binder::bind(outlives).to_predicate(tcx),
                ));
            }

            // `T` -> `Trait`
            (_, &ty::Dynamic(ref data, r)) => {
                let mut object_dids = data.auto_traits().chain(data.principal_def_id());
                if let Some(did) = object_dids.find(|did| !tcx.is_object_safe(*did)) {
                    return Err(TraitNotObjectSafe(did));
                }

                let cause = ObligationCause::new(
                    obligation.cause.span,
                    obligation.cause.body_id,
                    ObjectCastObligation(target),
                );

                let predicate_to_obligation = |predicate| {
                    Obligation::with_depth(
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
                let tr = ty::TraitRef::new(
                    tcx.require_lang_item(LangItem::Sized, None),
                    tcx.mk_substs_trait(source, &[]),
                );
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
                let maybe_unsizing_param_idx = |arg: GenericArg<'tcx>| match arg.unpack() {
                    GenericArgKind::Type(ty) => match ty.kind() {
                        ty::Param(p) => Some(p.index),
                        _ => None,
                    },

                    // Lifetimes aren't allowed to change during unsizing.
                    GenericArgKind::Lifetime(_) => None,

                    GenericArgKind::Const(ct) => match ct.val {
                        ty::ConstKind::Param(p) => Some(p.index),
                        _ => None,
                    },
                };

                // The last field of the structure has to exist and contain type/const parameters.
                let (tail_field, prefix_fields) =
                    def.non_enum_variant().fields.split_last().ok_or(Unimplemented)?;
                let tail_field_ty = tcx.type_of(tail_field.did);

                let mut unsizing_params = GrowableBitSet::new_empty();
                let mut found = false;
                for arg in tail_field_ty.walk() {
                    if let Some(i) = maybe_unsizing_param_idx(arg) {
                        unsizing_params.insert(i);
                        found = true;
                    }
                }
                if !found {
                    return Err(Unimplemented);
                }

                // Ensure none of the other fields mention the parameters used
                // in unsizing.
                // FIXME(eddyb) cache this (including computing `unsizing_params`)
                // by putting it in a query; it would only need the `DefId` as it
                // looks at declared field types, not anything substituted.
                for field in prefix_fields {
                    for arg in tcx.type_of(field.did).walk() {
                        if let Some(i) = maybe_unsizing_param_idx(arg) {
                            if unsizing_params.contains(i) {
                                return Err(Unimplemented);
                            }
                        }
                    }
                }

                // Extract `TailField<T>` and `TailField<U>` from `Struct<T>` and `Struct<U>`.
                let source_tail = tail_field_ty.subst(tcx, substs_a);
                let target_tail = tail_field_ty.subst(tcx, substs_b);

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
                    source_tail,
                    &[target_tail.into()],
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
                let new_tuple = tcx.mk_tup(
                    a_mid.iter().map(|k| k.expect_ty()).chain(iter::once(b_last.expect_ty())),
                );
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
                        a_last.expect_ty(),
                        &[b_last],
                    )
                }));
            }

            _ => bug!(),
        };

        Ok(ImplSourceBuiltinData { nested })
    }
}
