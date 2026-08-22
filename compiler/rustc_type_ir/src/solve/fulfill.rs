use std::cmp;
use std::hash::{Hash, Hasher};

use derive_where::derive_where;
use rustc_type_ir_macros::{GenericTypeVisitable, TypeFoldable_Generic, TypeVisitable_Generic};

use super::Goal;
use crate::{Interner, Upcast};

/// An `Obligation` represents some trait reference (e.g., `i32: Eq`) for
/// which the "impl_source" must be found. The process of finding an "impl_source" is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for i32`) that
/// satisfies the obligation, or else finding a bound that is in
/// scope.
#[derive_where(Clone, Debug; I: Interner, T)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
pub struct Obligation<I: Interner, T> {
    /// The reason we have to prove this thing.
    /// FIXME: we shouldn't ignore the cause but instead change the affected visitors
    /// to only visit predicates manually.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub cause: I::ObligationCause,

    /// The environment in which we should prove this thing.
    pub param_env: I::ParamEnv,

    /// The thing we are trying to prove.
    pub predicate: T,

    /// If we started proving this as a result of trying to prove
    /// something else, track the total depth to ensure termination.
    /// If this goes over a certain threshold, we abort compilation --
    /// in such cases, we can not say whether or not the predicate
    /// holds for certain. Stupid halting problem; such a drag.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub recursion_depth: usize,
}

impl<I: Interner, T: Copy> Obligation<I, T> {
    pub fn as_goal(&self) -> Goal<I, T> {
        Goal { param_env: self.param_env, predicate: self.predicate }
    }
}

impl<I: Interner, T: PartialEq> PartialEq for Obligation<I, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Ignore `cause` and `recursion_depth`. This is a small performance
        // win for a few crates, and a huge performance win for the crate in
        // https://github.com/rust-lang/rustc-perf/pull/1680, which greatly
        // stresses the trait system.
        self.param_env == other.param_env && self.predicate == other.predicate
    }
}

impl<I: Interner, T: Eq> Eq for Obligation<I, T> {}

impl<I: Interner, T: Hash> Hash for Obligation<I, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // See the comment on `Obligation::eq`.
        self.param_env.hash(state);
        self.predicate.hash(state);
    }
}

impl<I: Interner, O> Obligation<I, O> {
    pub fn new(
        interner: I,
        cause: I::ObligationCause,
        param_env: I::ParamEnv,
        predicate: impl Upcast<I, O>,
    ) -> Self {
        Self::with_depth(interner, cause, 0, param_env, predicate)
    }

    /// We often create nested obligations without setting the correct depth.
    ///
    /// To deal with this evaluate and fulfill explicitly update the depth
    /// of nested obligations using this function.
    pub fn set_depth_from_parent(&mut self, parent_depth: usize) {
        self.recursion_depth = cmp::max(parent_depth + 1, self.recursion_depth);
    }

    pub fn with_depth(
        interner: I,
        cause: I::ObligationCause,
        recursion_depth: usize,
        param_env: I::ParamEnv,
        predicate: impl Upcast<I, O>,
    ) -> Self {
        let predicate = predicate.upcast(interner);
        Obligation { cause, param_env, recursion_depth, predicate }
    }

    pub fn with<P>(&self, interner: I, value: impl Upcast<I, P>) -> Obligation<I, P> {
        Obligation::with_depth(
            interner,
            self.cause.clone(),
            self.recursion_depth,
            self.param_env,
            value,
        )
    }
}
