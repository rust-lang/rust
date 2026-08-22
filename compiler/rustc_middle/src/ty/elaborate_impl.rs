use rustc_span::Span;
use rustc_type_ir::elaborate::Elaboratable;
use rustc_type_ir::solve::Obligation;

use crate::traits::{ImplDerivedCause, ObligationCauseCode};
use crate::ty::{self, TyCtxt};

type PredicateObligation<'tcx> = Obligation<TyCtxt<'tcx>, ty::Predicate<'tcx>>;

/// For `Obligation`, a sub-obligation is combined with the current
/// obligation's param-env and cause code.
impl<'tcx> Elaboratable<TyCtxt<'tcx>> for PredicateObligation<'tcx> {
    fn predicate(&self) -> ty::Predicate<'tcx> {
        self.predicate
    }

    fn child(&self, clause: ty::Clause<'tcx>) -> Self {
        Obligation {
            cause: self.cause.clone(),
            param_env: self.param_env,
            recursion_depth: 0,
            predicate: clause.as_predicate(),
        }
    }

    fn child_with_derived_cause(
        &self,
        clause: ty::Clause<'tcx>,
        span: Span,
        parent_trait_pred: ty::PolyTraitClause<'tcx>,
        index: usize,
    ) -> Self {
        let cause = self.cause.clone().derived_cause(parent_trait_pred, |derived| {
            ObligationCauseCode::ImplDerived(Box::new(ImplDerivedCause {
                derived,
                impl_or_alias_def_id: parent_trait_pred.def_id(),
                impl_def_clause_index: Some(index),
                span,
            }))
        });

        Obligation {
            cause,
            param_env: self.param_env,
            recursion_depth: 0,
            predicate: clause.as_predicate(),
        }
    }
}

impl<'tcx> Elaboratable<TyCtxt<'tcx>> for ty::Clause<'tcx> {
    fn predicate(&self) -> ty::Predicate<'tcx> {
        self.as_predicate()
    }

    fn child(&self, clause: ty::Clause<'tcx>) -> Self {
        clause
    }

    fn child_with_derived_cause(
        &self,
        clause: ty::Clause<'tcx>,
        _span: Span,
        _parent_trait_pred: ty::PolyTraitClause<'tcx>,
        _index: usize,
    ) -> Self {
        clause
    }
}

impl<'tcx> Elaboratable<TyCtxt<'tcx>> for ty::Predicate<'tcx> {
    fn predicate(&self) -> ty::Predicate<'tcx> {
        *self
    }

    fn child(&self, clause: ty::Clause<'tcx>) -> Self {
        clause.as_predicate()
    }

    fn child_with_derived_cause(
        &self,
        clause: ty::Clause<'tcx>,
        _span: Span,
        _parent_trait_pred: ty::PolyTraitClause<'tcx>,
        _index: usize,
    ) -> Self {
        clause.as_predicate()
    }
}

impl<'tcx> Elaboratable<TyCtxt<'tcx>> for (ty::Predicate<'tcx>, Span) {
    fn predicate(&self) -> ty::Predicate<'tcx> {
        self.0
    }

    fn child(&self, clause: ty::Clause<'tcx>) -> Self {
        (clause.as_predicate(), self.1)
    }

    fn child_with_derived_cause(
        &self,
        clause: ty::Clause<'tcx>,
        _span: Span,
        _parent_trait_pred: ty::PolyTraitClause<'tcx>,
        _index: usize,
    ) -> Self {
        (clause.as_predicate(), self.1)
    }
}

impl<'tcx> Elaboratable<TyCtxt<'tcx>> for (ty::Clause<'tcx>, Span) {
    fn predicate(&self) -> ty::Predicate<'tcx> {
        self.0.as_predicate()
    }

    fn child(&self, clause: ty::Clause<'tcx>) -> Self {
        (clause, self.1)
    }

    fn child_with_derived_cause(
        &self,
        clause: ty::Clause<'tcx>,
        _span: Span,
        _parent_trait_pred: ty::PolyTraitClause<'tcx>,
        _index: usize,
    ) -> Self {
        (clause, self.1)
    }
}
