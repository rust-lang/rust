use rustc_hir::def_id::DefId;
use rustc_infer::infer::at::ToTrace;
use rustc_infer::infer::canonical::CanonicalVarValues;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{InferCtxt, InferOk, LateBoundRegionConversionTime};
use rustc_infer::traits::query::NoSolution;
use rustc_infer::traits::ObligationCause;
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor,
};
use rustc_span::DUMMY_SP;
use std::ops::ControlFlow;

use super::search_graph::SearchGraph;
use super::Goal;

pub struct EvalCtxt<'a, 'tcx> {
    // FIXME: should be private.
    pub(super) infcx: &'a InferCtxt<'tcx>,
    pub(super) var_values: CanonicalVarValues<'tcx>,
    /// The highest universe index nameable by the caller.
    ///
    /// When we enter a new binder inside of the query we create new universes
    /// which the caller cannot name. We have to be careful with variables from
    /// these new universes when creating the query response.
    ///
    /// Both because these new universes can prevent us from reaching a fixpoint
    /// if we have a coinductive cycle and because that's the only way we can return
    /// new placeholders to the caller.
    pub(super) max_input_universe: ty::UniverseIndex,

    pub(super) search_graph: &'a mut SearchGraph<'tcx>,

    /// This field is used by a debug assertion in [`EvalCtxt::evaluate_goal`],
    /// see the comment in that method for more details.
    pub in_projection_eq_hack: bool,
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    pub(super) fn probe<T>(&mut self, f: impl FnOnce(&mut EvalCtxt<'_, 'tcx>) -> T) -> T {
        self.infcx.probe(|_| f(self))
    }

    pub(super) fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    pub(super) fn next_ty_infer(&self) -> Ty<'tcx> {
        self.infcx.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: DUMMY_SP,
        })
    }

    pub(super) fn next_const_infer(&self, ty: Ty<'tcx>) -> ty::Const<'tcx> {
        self.infcx.next_const_var(
            ty,
            ConstVariableOrigin { kind: ConstVariableOriginKind::MiscVariable, span: DUMMY_SP },
        )
    }

    /// Is the projection predicate is of the form `exists<T> <Ty as Trait>::Assoc = T`.
    ///
    /// This is the case if the `term` is an inference variable in the innermost universe
    /// and does not occur in any other part of the predicate.
    pub(super) fn term_is_fully_unconstrained(
        &self,
        goal: Goal<'tcx, ty::ProjectionPredicate<'tcx>>,
    ) -> bool {
        let term_is_infer = match goal.predicate.term.unpack() {
            ty::TermKind::Ty(ty) => {
                if let &ty::Infer(ty::TyVar(vid)) = ty.kind() {
                    match self.infcx.probe_ty_var(vid) {
                        Ok(value) => bug!("resolved var in query: {goal:?} {value:?}"),
                        Err(universe) => universe == self.universe(),
                    }
                } else {
                    false
                }
            }
            ty::TermKind::Const(ct) => {
                if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
                    match self.infcx.probe_const_var(vid) {
                        Ok(value) => bug!("resolved var in query: {goal:?} {value:?}"),
                        Err(universe) => universe == self.universe(),
                    }
                } else {
                    false
                }
            }
        };

        // Guard against `<T as Trait<?0>>::Assoc = ?0>`.
        struct ContainsTerm<'tcx> {
            term: ty::Term<'tcx>,
        }
        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ContainsTerm<'tcx> {
            type BreakTy = ();
            fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                if t.needs_infer() {
                    if ty::Term::from(t) == self.term {
                        ControlFlow::Break(())
                    } else {
                        t.super_visit_with(self)
                    }
                } else {
                    ControlFlow::Continue(())
                }
            }

            fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
                if c.needs_infer() {
                    if ty::Term::from(c) == self.term {
                        ControlFlow::Break(())
                    } else {
                        c.super_visit_with(self)
                    }
                } else {
                    ControlFlow::Continue(())
                }
            }
        }

        let mut visitor = ContainsTerm { term: goal.predicate.term };

        term_is_infer
            && goal.predicate.projection_ty.visit_with(&mut visitor).is_continue()
            && goal.param_env.visit_with(&mut visitor).is_continue()
    }

    #[instrument(level = "debug", skip(self, param_env), ret)]
    pub(super) fn eq<T: ToTrace<'tcx>>(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution> {
        self.infcx
            .at(&ObligationCause::dummy(), param_env)
            .eq(lhs, rhs)
            .map(|InferOk { value: (), obligations }| {
                obligations.into_iter().map(|o| o.into()).collect()
            })
            .map_err(|e| {
                debug!(?e, "failed to equate");
                NoSolution
            })
    }

    pub(super) fn instantiate_binder_with_infer<T: TypeFoldable<TyCtxt<'tcx>> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T {
        self.infcx.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::HigherRankedType,
            value,
        )
    }

    pub(super) fn instantiate_binder_with_placeholders<T: TypeFoldable<TyCtxt<'tcx>> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T {
        self.infcx.instantiate_binder_with_placeholders(value)
    }

    pub(super) fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.infcx.resolve_vars_if_possible(value)
    }

    pub(super) fn fresh_substs_for_item(&self, def_id: DefId) -> ty::SubstsRef<'tcx> {
        self.infcx.fresh_substs_for_item(DUMMY_SP, def_id)
    }

    pub(super) fn universe(&self) -> ty::UniverseIndex {
        self.infcx.universe()
    }
}
