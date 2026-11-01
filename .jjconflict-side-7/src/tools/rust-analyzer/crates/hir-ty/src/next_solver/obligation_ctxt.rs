use hir_def::TraitId;
use rustc_type_ir::{TypeFoldable, Upcast, Variance};

use crate::next_solver::{
    Const, DbInterner, ParamEnv, Term, TraitRef, Ty, TypeError,
    fulfill::{FulfillmentCtxt, NextSolverError},
    infer::{
        InferCtxt, InferOk,
        at::ToTrace,
        traits::{Obligation, ObligationCause, PredicateObligation, PredicateObligations},
    },
};

/// Used if you want to have pleasant experience when dealing
/// with obligations outside of hir or mir typeck.
pub struct ObligationCtxt<'a, 'db> {
    pub infcx: &'a InferCtxt<'db>,
    engine: FulfillmentCtxt<'db>,
}

impl<'a, 'db> ObligationCtxt<'a, 'db> {
    pub fn new(infcx: &'a InferCtxt<'db>) -> Self {
        Self { infcx, engine: FulfillmentCtxt::new(infcx) }
    }
}

impl<'a, 'db> ObligationCtxt<'a, 'db> {
    pub fn register_obligation(&mut self, obligation: PredicateObligation<'db>) {
        self.engine.register_predicate_obligation(self.infcx, obligation);
    }

    pub fn register_obligations(
        &mut self,
        obligations: impl IntoIterator<Item = PredicateObligation<'db>>,
    ) {
        self.engine.register_predicate_obligations(self.infcx, obligations);
    }

    pub fn register_infer_ok_obligations<T>(&mut self, infer_ok: InferOk<'db, T>) -> T {
        let InferOk { value, obligations } = infer_ok;
        self.register_obligations(obligations);
        value
    }

    /// Requires that `ty` must implement the trait with `def_id` in
    /// the given environment. This trait must not have any type
    /// parameters (except for `Self`).
    pub fn register_bound(
        &mut self,
        cause: ObligationCause,
        param_env: ParamEnv<'db>,
        ty: Ty<'db>,
        def_id: TraitId,
    ) {
        let trait_ref = TraitRef::new(self.infcx.interner, def_id.into(), [ty]);
        self.register_obligation(Obligation {
            cause,
            recursion_depth: 0,
            param_env,
            predicate: trait_ref.upcast(self.infcx.interner),
        });
    }

    pub fn eq<T: ToTrace<'db>>(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'db>> {
        self.infcx
            .at(cause, param_env)
            .eq(expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    /// Checks whether `expected` is a subtype of `actual`: `expected <: actual`.
    pub fn sub<T: ToTrace<'db>>(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'db>> {
        self.infcx
            .at(cause, param_env)
            .sub(expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    pub fn relate<T: ToTrace<'db>>(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        variance: Variance,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'db>> {
        self.infcx
            .at(cause, param_env)
            .relate(expected, variance, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    /// Checks whether `expected` is a supertype of `actual`: `expected :> actual`.
    pub fn sup<T: ToTrace<'db>>(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        expected: T,
        actual: T,
    ) -> Result<(), TypeError<'db>> {
        self.infcx
            .at(cause, param_env)
            .sup(expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    /// Computes the least-upper-bound, or mutual supertype, of two values.
    pub fn lub<T: ToTrace<'db>>(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        expected: T,
        actual: T,
    ) -> Result<T, TypeError<'db>> {
        self.infcx
            .at(cause, param_env)
            .lub(expected, actual)
            .map(|infer_ok| self.register_infer_ok_obligations(infer_ok))
    }

    #[must_use]
    pub fn try_evaluate_obligations(&mut self) -> Vec<NextSolverError<'db>> {
        self.engine.try_evaluate_obligations(self.infcx)
    }

    #[must_use]
    pub fn evaluate_obligations_error_on_ambiguity(&mut self) -> Vec<NextSolverError<'db>> {
        self.engine.evaluate_obligations_error_on_ambiguity(self.infcx)
    }

    /// Returns the not-yet-processed and stalled obligations from the
    /// `ObligationCtxt`.
    ///
    /// Takes ownership of the context as doing operations such as
    /// [`ObligationCtxt::eq`] afterwards will result in other obligations
    /// getting ignored. You can make a new `ObligationCtxt` if this
    /// needs to be done in a loop, for example.
    #[must_use]
    pub fn into_pending_obligations(self) -> PredicateObligations<'db> {
        self.engine.pending_obligations()
    }

    pub fn deeply_normalize<T: TypeFoldable<DbInterner<'db>>>(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        value: T,
    ) -> Result<T, Vec<NextSolverError<'db>>> {
        self.infcx.at(cause, param_env).deeply_normalize(value)
    }

    pub fn structurally_normalize_ty(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        value: Ty<'db>,
    ) -> Result<Ty<'db>, Vec<NextSolverError<'db>>> {
        self.infcx.at(cause, param_env).structurally_normalize_ty(value, &mut self.engine)
    }

    pub fn structurally_normalize_const(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        value: Const<'db>,
    ) -> Result<Const<'db>, Vec<NextSolverError<'db>>> {
        self.infcx.at(cause, param_env).structurally_normalize_const(value, &mut self.engine)
    }

    pub fn structurally_normalize_term(
        &mut self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        value: Term<'db>,
    ) -> Result<Term<'db>, Vec<NextSolverError<'db>>> {
        self.infcx.at(cause, param_env).structurally_normalize_term(value, &mut self.engine)
    }
}
