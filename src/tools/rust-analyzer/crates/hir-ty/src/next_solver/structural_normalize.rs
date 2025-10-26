use rustc_type_ir::{AliasRelationDirection, inherent::Term as _};

use crate::next_solver::{
    Const, PredicateKind, Term, Ty,
    fulfill::{FulfillmentCtxt, NextSolverError},
    infer::{at::At, traits::Obligation},
};

impl<'db> At<'_, 'db> {
    pub(crate) fn structurally_normalize_ty(
        &self,
        ty: Ty<'db>,
        fulfill_cx: &mut FulfillmentCtxt<'db>,
    ) -> Result<Ty<'db>, Vec<NextSolverError<'db>>> {
        self.structurally_normalize_term(ty.into(), fulfill_cx).map(|term| term.expect_type())
    }

    pub(crate) fn structurally_normalize_const(
        &self,
        ct: Const<'db>,
        fulfill_cx: &mut FulfillmentCtxt<'db>,
    ) -> Result<Const<'db>, Vec<NextSolverError<'db>>> {
        self.structurally_normalize_term(ct.into(), fulfill_cx).map(|term| term.expect_const())
    }

    pub(crate) fn structurally_normalize_term(
        &self,
        term: Term<'db>,
        fulfill_cx: &mut FulfillmentCtxt<'db>,
    ) -> Result<Term<'db>, Vec<NextSolverError<'db>>> {
        assert!(!term.is_infer(), "should have resolved vars before calling");

        if term.to_alias_term().is_none() {
            return Ok(term);
        }

        let new_infer = self.infcx.next_term_var_of_kind(term);

        // We simply emit an `alias-eq` goal here, since that will take care of
        // normalizing the LHS of the projection until it is a rigid projection
        // (or a not-yet-defined opaque in scope).
        let obligation = Obligation::new(
            self.infcx.interner,
            self.cause.clone(),
            self.param_env,
            PredicateKind::AliasRelate(term, new_infer, AliasRelationDirection::Equate),
        );

        fulfill_cx.register_predicate_obligation(self.infcx, obligation);
        let errors = fulfill_cx.try_evaluate_obligations(self.infcx);
        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(self.infcx.resolve_vars_if_possible(new_infer))
    }
}
