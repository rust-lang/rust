//! Defining `SolverContext` for next-trait-solver.

use hir_def::{AssocItemId, GeneralConstId};
use rustc_next_trait_solver::delegate::SolverDelegate;
use rustc_type_ir::{
    AliasTyKind, GenericArgKind, InferCtxtLike, Interner, PredicatePolarity, TypeFlags,
    TypeVisitableExt,
    inherent::{IntoKind, Term as _, Ty as _},
    lang_items::SolverTraitLangItem,
    solve::{Certainty, NoSolution},
};
use tracing::debug;

use crate::next_solver::{
    AliasTy, AnyImplId, CanonicalVarKind, Clause, ClauseKind, CoercePredicate, GenericArgs,
    ParamEnv, Predicate, PredicateKind, SubtypePredicate, Ty, TyKind, fold::fold_tys,
    util::sizedness_fast_path,
};

use super::{
    DbInterner, ErrorGuaranteed, GenericArg, SolverDefId, Span,
    infer::{DbInternerInferExt, InferCtxt, canonical::instantiate::CanonicalExt},
};

pub type Goal<'db, P> = rustc_type_ir::solve::Goal<DbInterner<'db>, P>;

#[repr(transparent)]
pub(crate) struct SolverContext<'db>(pub(crate) InferCtxt<'db>);

impl<'a, 'db> From<&'a InferCtxt<'db>> for &'a SolverContext<'db> {
    fn from(infcx: &'a InferCtxt<'db>) -> Self {
        // SAFETY: `repr(transparent)`
        unsafe { std::mem::transmute(infcx) }
    }
}

impl<'db> std::ops::Deref for SolverContext<'db> {
    type Target = InferCtxt<'db>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'db> SolverDelegate for SolverContext<'db> {
    type Interner = DbInterner<'db>;
    type Infcx = InferCtxt<'db>;

    fn cx(&self) -> Self::Interner {
        self.0.interner
    }

    fn build_with_canonical<V>(
        cx: Self::Interner,
        canonical: &rustc_type_ir::CanonicalQueryInput<Self::Interner, V>,
    ) -> (Self, V, rustc_type_ir::CanonicalVarValues<Self::Interner>)
    where
        V: rustc_type_ir::TypeFoldable<Self::Interner>,
    {
        let (infcx, value, vars) = cx.infer_ctxt().build_with_canonical(canonical);
        (SolverContext(infcx), value, vars)
    }

    fn fresh_var_for_kind_with_span(&self, arg: GenericArg<'db>, _span: Span) -> GenericArg<'db> {
        match arg.kind() {
            GenericArgKind::Lifetime(_) => self.next_region_var().into(),
            GenericArgKind::Type(_) => self.next_ty_var().into(),
            GenericArgKind::Const(_) => self.next_const_var().into(),
        }
    }

    fn leak_check(
        &self,
        _max_input_universe: rustc_type_ir::UniverseIndex,
    ) -> Result<(), NoSolution> {
        Ok(())
    }

    fn well_formed_goals(
        &self,
        _param_env: ParamEnv<'db>,
        _arg: <Self::Interner as rustc_type_ir::Interner>::Term,
    ) -> Option<
        Vec<
            rustc_type_ir::solve::Goal<
                Self::Interner,
                <Self::Interner as rustc_type_ir::Interner>::Predicate,
            >,
        >,
    > {
        // FIXME(next-solver):
        None
    }

    fn make_deduplicated_outlives_constraints(
        &self,
    ) -> Vec<
        rustc_type_ir::OutlivesPredicate<
            Self::Interner,
            <Self::Interner as rustc_type_ir::Interner>::GenericArg,
        >,
    > {
        // FIXME: add if we care about regions
        vec![]
    }

    fn instantiate_canonical<V>(
        &self,
        canonical: rustc_type_ir::Canonical<Self::Interner, V>,
        values: rustc_type_ir::CanonicalVarValues<Self::Interner>,
    ) -> V
    where
        V: rustc_type_ir::TypeFoldable<Self::Interner>,
    {
        canonical.instantiate(self.cx(), &values)
    }

    fn instantiate_canonical_var(
        &self,
        kind: CanonicalVarKind<'db>,
        _span: <Self::Interner as Interner>::Span,
        var_values: &[GenericArg<'db>],
        universe_map: impl Fn(rustc_type_ir::UniverseIndex) -> rustc_type_ir::UniverseIndex,
    ) -> GenericArg<'db> {
        self.0.instantiate_canonical_var(kind, var_values, universe_map)
    }

    fn add_item_bounds_for_hidden_type(
        &self,
        def_id: SolverDefId,
        args: GenericArgs<'db>,
        param_env: ParamEnv<'db>,
        hidden_ty: Ty<'db>,
        goals: &mut Vec<Goal<'db, Predicate<'db>>>,
    ) {
        let interner = self.interner;
        let opaque_id = def_id.expect_opaque_ty();
        // Require that the hidden type is well-formed. We have to
        // make sure we wf-check the hidden type to fix #114728.
        //
        // However, we don't check that all types are well-formed.
        // We only do so for types provided by the user or if they are
        // "used", e.g. for method selection.
        //
        // This means we never check the wf requirements of the hidden
        // type during MIR borrowck, causing us to infer the wrong
        // lifetime for its member constraints which then results in
        // unexpected region errors.
        goals.push(Goal::new(interner, param_env, ClauseKind::WellFormed(hidden_ty.into())));

        let replace_opaques_in = |clause: Clause<'db>| {
            fold_tys(interner, clause, |ty| match ty.kind() {
                // Replace all other mentions of the same opaque type with the hidden type,
                // as the bounds must hold on the hidden type after all.
                TyKind::Alias(
                    AliasTyKind::Opaque,
                    AliasTy { def_id: def_id2, args: args2, .. },
                ) if def_id == def_id2 && args == args2 => hidden_ty,
                _ => ty,
            })
        };

        let item_bounds = opaque_id.predicates(interner.db);
        for predicate in item_bounds.iter_instantiated_copied(interner, args.as_slice()) {
            let predicate = replace_opaques_in(predicate);

            // Require that the predicate holds for the concrete type.
            debug!(?predicate);
            goals.push(Goal::new(interner, param_env, predicate));
        }
    }

    fn fetch_eligible_assoc_item(
        &self,
        _goal_trait_ref: rustc_type_ir::TraitRef<Self::Interner>,
        trait_assoc_def_id: SolverDefId,
        impl_id: AnyImplId,
    ) -> Result<Option<SolverDefId>, ErrorGuaranteed> {
        let AnyImplId::ImplId(impl_id) = impl_id else {
            // Builtin derive traits don't have type/consts assoc items.
            return Ok(None);
        };
        let impl_items = impl_id.impl_items(self.0.interner.db());
        let id =
            match trait_assoc_def_id {
                SolverDefId::TypeAliasId(trait_assoc_id) => {
                    let trait_assoc_data = self.0.interner.db.type_alias_signature(trait_assoc_id);
                    impl_items
                        .items
                        .iter()
                        .find_map(|(impl_assoc_name, impl_assoc_id)| {
                            if let AssocItemId::TypeAliasId(impl_assoc_id) = *impl_assoc_id
                                && *impl_assoc_name == trait_assoc_data.name
                            {
                                Some(impl_assoc_id)
                            } else {
                                None
                            }
                        })
                        .or_else(|| {
                            if trait_assoc_data.ty.is_some() { Some(trait_assoc_id) } else { None }
                        })
                        .map(SolverDefId::TypeAliasId)
                }
                SolverDefId::ConstId(trait_assoc_id) => {
                    let trait_assoc_data = self.0.interner.db.const_signature(trait_assoc_id);
                    let trait_assoc_name = trait_assoc_data
                        .name
                        .as_ref()
                        .expect("unnamed consts should not get passed to the solver");
                    impl_items
                        .items
                        .iter()
                        .find_map(|(impl_assoc_name, impl_assoc_id)| {
                            if let AssocItemId::ConstId(impl_assoc_id) = *impl_assoc_id
                                && impl_assoc_name == trait_assoc_name
                            {
                                Some(impl_assoc_id)
                            } else {
                                None
                            }
                        })
                        .or_else(|| {
                            if trait_assoc_data.has_body() { Some(trait_assoc_id) } else { None }
                        })
                        .map(SolverDefId::ConstId)
                }
                _ => panic!("Unexpected SolverDefId"),
            };
        Ok(id)
    }

    fn is_transmutable(
        &self,
        _src: Ty<'db>,
        _dst: Ty<'db>,
        _assume: <Self::Interner as rustc_type_ir::Interner>::Const,
    ) -> Result<Certainty, NoSolution> {
        // It's better to return some value while not fully implement
        // then panic in the mean time
        Ok(Certainty::Yes)
    }

    fn evaluate_const(
        &self,
        _param_env: ParamEnv<'db>,
        uv: rustc_type_ir::UnevaluatedConst<Self::Interner>,
    ) -> Option<<Self::Interner as rustc_type_ir::Interner>::Const> {
        match uv.def.0 {
            GeneralConstId::ConstId(c) => {
                let subst = uv.args;
                let ec = self.cx().db.const_eval(c, subst, None).ok()?;
                Some(ec)
            }
            GeneralConstId::StaticId(c) => {
                let ec = self.cx().db.const_eval_static(c).ok()?;
                Some(ec)
            }
        }
    }

    fn compute_goal_fast_path(
        &self,
        goal: rustc_type_ir::solve::Goal<
            Self::Interner,
            <Self::Interner as rustc_type_ir::Interner>::Predicate,
        >,
        _span: <Self::Interner as rustc_type_ir::Interner>::Span,
    ) -> Option<Certainty> {
        if let Some(trait_pred) = goal.predicate.as_trait_clause() {
            if self.shallow_resolve(trait_pred.self_ty().skip_binder()).is_ty_var()
                // We don't do this fast path when opaques are defined since we may
                // eventually use opaques to incompletely guide inference via ty var
                // self types.
                // FIXME: Properly consider opaques here.
                && self.inner.borrow_mut().opaque_types().is_empty()
            {
                return Some(Certainty::AMBIGUOUS);
            }

            if trait_pred.polarity() == PredicatePolarity::Positive {
                match self.0.cx().as_trait_lang_item(trait_pred.def_id()) {
                    Some(SolverTraitLangItem::Sized) | Some(SolverTraitLangItem::MetaSized) => {
                        let predicate = self.resolve_vars_if_possible(goal.predicate);
                        if sizedness_fast_path(self.cx(), predicate, goal.param_env) {
                            return Some(Certainty::Yes);
                        }
                    }
                    Some(SolverTraitLangItem::Copy | SolverTraitLangItem::Clone) => {
                        let self_ty =
                            self.resolve_vars_if_possible(trait_pred.self_ty().skip_binder());
                        // Unlike `Sized` traits, which always prefer the built-in impl,
                        // `Copy`/`Clone` may be shadowed by a param-env candidate which
                        // could force a lifetime error or guide inference. While that's
                        // not generally desirable, it is observable, so for now let's
                        // ignore this fast path for types that have regions or infer.
                        if !self_ty
                            .has_type_flags(TypeFlags::HAS_FREE_REGIONS | TypeFlags::HAS_INFER)
                            && self_ty.is_trivially_pure_clone_copy()
                        {
                            return Some(Certainty::Yes);
                        }
                    }
                    _ => {}
                }
            }
        }

        let pred = goal.predicate.kind();
        match pred.no_bound_vars()? {
            PredicateKind::Clause(ClauseKind::RegionOutlives(_outlives)) => Some(Certainty::Yes),
            PredicateKind::Clause(ClauseKind::TypeOutlives(_outlives)) => Some(Certainty::Yes),
            PredicateKind::Subtype(SubtypePredicate { a, b, .. })
            | PredicateKind::Coerce(CoercePredicate { a, b }) => {
                if self.shallow_resolve(a).is_ty_var() && self.shallow_resolve(b).is_ty_var() {
                    // FIXME: We also need to register a subtype relation between these vars
                    // when those are added, and if they aren't in the same sub root then
                    // we should mark this goal as `has_changed`.
                    Some(Certainty::AMBIGUOUS)
                } else {
                    None
                }
            }
            PredicateKind::Clause(ClauseKind::ConstArgHasType(ct, _)) => {
                if self.shallow_resolve_const(ct).is_ct_infer() {
                    Some(Certainty::AMBIGUOUS)
                } else {
                    None
                }
            }
            PredicateKind::Clause(ClauseKind::WellFormed(arg)) => {
                if arg.is_trivially_wf(self.interner) {
                    Some(Certainty::Yes)
                } else if arg.is_infer() {
                    Some(Certainty::AMBIGUOUS)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
