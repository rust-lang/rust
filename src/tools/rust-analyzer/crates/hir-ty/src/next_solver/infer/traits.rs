//! Trait Resolution. See the [rustc-dev-guide] for more information on how this works.
//!
//! [rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

use std::{
    cmp,
    hash::{Hash, Hasher},
};

use rustc_type_ir::{
    PredicatePolarity, Upcast,
    solve::{Certainty, NoSolution},
};

use crate::next_solver::{
    Binder, DbInterner, Goal, ParamEnv, PolyTraitPredicate, Predicate, SolverDefId, TraitPredicate,
    Ty,
};

use super::InferCtxt;

/// The reason why we incurred this obligation; used for error reporting.
///
/// Non-misc `ObligationCauseCode`s are stored on the heap. This gives the
/// best trade-off between keeping the type small (which makes copies cheaper)
/// while not doing too many heap allocations.
///
/// We do not want to intern this as there are a lot of obligation causes which
/// only live for a short period of time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObligationCause {
    /// The ID of the fn body that triggered this obligation. This is
    /// used for region obligations to determine the precise
    /// environment in which the region obligation should be evaluated
    /// (in particular, closures can add new assumptions). See the
    /// field `region_obligations` of the `FulfillmentContext` for more
    /// information.
    pub body_id: Option<SolverDefId>,
}

impl ObligationCause {
    #[inline]
    pub fn new(body_id: SolverDefId) -> ObligationCause {
        ObligationCause { body_id: Some(body_id) }
    }

    #[inline(always)]
    pub fn dummy() -> ObligationCause {
        ObligationCause { body_id: None }
    }
}

/// An `Obligation` represents some trait reference (e.g., `i32: Eq`) for
/// which the "impl_source" must be found. The process of finding an "impl_source" is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for i32`) that
/// satisfies the obligation, or else finding a bound that is in
/// scope. The eventual result is usually a `Selection` (defined below).
#[derive(Clone, Debug)]
pub struct Obligation<'db, T> {
    /// The reason we have to prove this thing.
    pub cause: ObligationCause,

    /// The environment in which we should prove this thing.
    pub param_env: ParamEnv<'db>,

    /// The thing we are trying to prove.
    pub predicate: T,

    /// If we started proving this as a result of trying to prove
    /// something else, track the total depth to ensure termination.
    /// If this goes over a certain threshold, we abort compilation --
    /// in such cases, we can not say whether or not the predicate
    /// holds for certain. Stupid halting problem; such a drag.
    pub recursion_depth: usize,
}

impl<'db, T: Copy> Obligation<'db, T> {
    pub fn as_goal(&self) -> Goal<'db, T> {
        Goal { param_env: self.param_env, predicate: self.predicate }
    }
}

impl<'db, T: PartialEq> PartialEq<Obligation<'db, T>> for Obligation<'db, T> {
    #[inline]
    fn eq(&self, other: &Obligation<'db, T>) -> bool {
        // Ignore `cause` and `recursion_depth`. This is a small performance
        // win for a few crates, and a huge performance win for the crate in
        // https://github.com/rust-lang/rustc-perf/pull/1680, which greatly
        // stresses the trait system.
        self.param_env == other.param_env && self.predicate == other.predicate
    }
}

impl<'db, T: Eq> Eq for Obligation<'db, T> {}

impl<'db, T: Hash> Hash for Obligation<'db, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // See the comment on `Obligation::eq`.
        self.param_env.hash(state);
        self.predicate.hash(state);
    }
}

impl<'db, P> From<Obligation<'db, P>> for Goal<'db, P> {
    fn from(value: Obligation<'db, P>) -> Self {
        Goal { param_env: value.param_env, predicate: value.predicate }
    }
}

pub type PredicateObligation<'db> = Obligation<'db, Predicate<'db>>;
pub type TraitObligation<'db> = Obligation<'db, TraitPredicate<'db>>;

pub type PredicateObligations<'db> = Vec<PredicateObligation<'db>>;

impl<'db> PredicateObligation<'db> {
    /// Flips the polarity of the inner predicate.
    ///
    /// Given `T: Trait` predicate it returns `T: !Trait` and given `T: !Trait` returns `T: Trait`.
    pub fn flip_polarity(&self, tcx: DbInterner<'db>) -> Option<PredicateObligation<'db>> {
        Some(PredicateObligation {
            cause: self.cause.clone(),
            param_env: self.param_env,
            predicate: self.predicate.flip_polarity()?,
            recursion_depth: self.recursion_depth,
        })
    }
}

impl<'db, O> Obligation<'db, O> {
    pub fn new(
        tcx: DbInterner<'db>,
        cause: ObligationCause,
        param_env: ParamEnv<'db>,
        predicate: impl Upcast<DbInterner<'db>, O>,
    ) -> Obligation<'db, O> {
        Self::with_depth(tcx, cause, 0, param_env, predicate)
    }

    /// We often create nested obligations without setting the correct depth.
    ///
    /// To deal with this evaluate and fulfill explicitly update the depth
    /// of nested obligations using this function.
    pub fn set_depth_from_parent(&mut self, parent_depth: usize) {
        self.recursion_depth = cmp::max(parent_depth + 1, self.recursion_depth);
    }

    pub fn with_depth(
        tcx: DbInterner<'db>,
        cause: ObligationCause,
        recursion_depth: usize,
        param_env: ParamEnv<'db>,
        predicate: impl Upcast<DbInterner<'db>, O>,
    ) -> Obligation<'db, O> {
        let predicate = predicate.upcast(tcx);
        Obligation { cause, param_env, recursion_depth, predicate }
    }

    pub fn misc(
        tcx: DbInterner<'db>,
        body_id: SolverDefId,
        param_env: ParamEnv<'db>,
        trait_ref: impl Upcast<DbInterner<'db>, O>,
    ) -> Obligation<'db, O> {
        Obligation::new(tcx, ObligationCause::new(body_id), param_env, trait_ref)
    }

    pub fn with<P>(
        &self,
        tcx: DbInterner<'db>,
        value: impl Upcast<DbInterner<'db>, P>,
    ) -> Obligation<'db, P> {
        Obligation::with_depth(tcx, self.cause.clone(), self.recursion_depth, self.param_env, value)
    }
}
