//! Defining `SolverContext` for next-trait-solver.

use hir_def::{AssocItemId, GeneralConstId, TypeAliasId};
use rustc_next_trait_solver::delegate::SolverDelegate;
use rustc_type_ir::GenericArgKind;
use rustc_type_ir::lang_items::SolverTraitLangItem;
use rustc_type_ir::{
    InferCtxtLike, Interner, PredicatePolarity, TypeFlags, TypeVisitableExt, UniverseIndex,
    inherent::{IntoKind, SliceLike, Span as _, Term as _, Ty as _},
    solve::{Certainty, NoSolution},
};

use crate::next_solver::mapping::NextSolverToChalk;
use crate::next_solver::{CanonicalVarKind, ImplIdWrapper};
use crate::{
    TraitRefExt,
    db::HirDatabase,
    next_solver::{
        ClauseKind, CoercePredicate, PredicateKind, SubtypePredicate, mapping::ChalkToNextSolver,
        util::sizedness_fast_path,
    },
};

use super::{
    Canonical, CanonicalVarValues, Const, DbInterner, ErrorGuaranteed, GenericArg, GenericArgs,
    ParamEnv, Predicate, SolverDefId, Span, Ty, UnevaluatedConst,
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

    fn fresh_var_for_kind_with_span(&self, arg: GenericArg<'db>, span: Span) -> GenericArg<'db> {
        match arg.kind() {
            GenericArgKind::Lifetime(_) => self.next_region_var().into(),
            GenericArgKind::Type(_) => self.next_ty_var().into(),
            GenericArgKind::Const(_) => self.next_const_var().into(),
        }
    }

    fn leak_check(
        &self,
        max_input_universe: rustc_type_ir::UniverseIndex,
    ) -> Result<(), NoSolution> {
        Ok(())
    }

    fn well_formed_goals(
        &self,
        param_env: <Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        arg: <Self::Interner as rustc_type_ir::Interner>::Term,
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
        span: <Self::Interner as Interner>::Span,
        var_values: &[GenericArg<'db>],
        universe_map: impl Fn(rustc_type_ir::UniverseIndex) -> rustc_type_ir::UniverseIndex,
    ) -> GenericArg<'db> {
        self.0.instantiate_canonical_var(kind, var_values, universe_map)
    }

    fn add_item_bounds_for_hidden_type(
        &self,
        def_id: <Self::Interner as rustc_type_ir::Interner>::DefId,
        args: <Self::Interner as rustc_type_ir::Interner>::GenericArgs,
        param_env: <Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        hidden_ty: <Self::Interner as rustc_type_ir::Interner>::Ty,
        goals: &mut Vec<
            rustc_type_ir::solve::Goal<
                Self::Interner,
                <Self::Interner as rustc_type_ir::Interner>::Predicate,
            >,
        >,
    ) {
        unimplemented!()
    }

    fn fetch_eligible_assoc_item(
        &self,
        goal_trait_ref: rustc_type_ir::TraitRef<Self::Interner>,
        trait_assoc_def_id: SolverDefId,
        impl_id: ImplIdWrapper,
    ) -> Result<Option<SolverDefId>, ErrorGuaranteed> {
        let trait_ = self
            .0
            .interner
            .db()
            .impl_trait(impl_id.0)
            // ImplIds for impls where the trait ref can't be resolved should never reach solver
            .expect("invalid impl passed to next-solver")
            .skip_binder()
            .def_id
            .0;
        let trait_data = trait_.trait_items(self.0.interner.db());
        let impl_items = impl_id.0.impl_items(self.0.interner.db());
        let id = match trait_assoc_def_id {
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
                    .map(SolverDefId::ConstId)
            }
            _ => panic!("Unexpected SolverDefId"),
        };
        Ok(id)
    }

    fn is_transmutable(
        &self,
        dst: <Self::Interner as rustc_type_ir::Interner>::Ty,
        src: <Self::Interner as rustc_type_ir::Interner>::Ty,
        assume: <Self::Interner as rustc_type_ir::Interner>::Const,
    ) -> Result<Certainty, NoSolution> {
        unimplemented!()
    }

    fn evaluate_const(
        &self,
        param_env: <Self::Interner as rustc_type_ir::Interner>::ParamEnv,
        uv: rustc_type_ir::UnevaluatedConst<Self::Interner>,
    ) -> Option<<Self::Interner as rustc_type_ir::Interner>::Const> {
        let c = match uv.def {
            SolverDefId::ConstId(c) => GeneralConstId::ConstId(c),
            SolverDefId::StaticId(c) => GeneralConstId::StaticId(c),
            _ => unreachable!(),
        };
        let subst = uv.args;
        let ec = self.cx().db.const_eval(c, subst, None).ok()?;
        Some(ec)
    }

    fn compute_goal_fast_path(
        &self,
        goal: rustc_type_ir::solve::Goal<
            Self::Interner,
            <Self::Interner as rustc_type_ir::Interner>::Predicate,
        >,
        span: <Self::Interner as rustc_type_ir::Interner>::Span,
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
            PredicateKind::Clause(ClauseKind::RegionOutlives(outlives)) => Some(Certainty::Yes),
            PredicateKind::Clause(ClauseKind::TypeOutlives(outlives)) => Some(Certainty::Yes),
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
