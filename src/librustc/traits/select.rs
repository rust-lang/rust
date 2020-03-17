//! Candidate selection. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html#selection

use self::EvaluationResult::*;

use super::{SelectionError, SelectionResult};

use crate::dep_graph::DepNodeIndex;
use crate::ty::{self, TyCtxt};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use rustc_hir::def_id::DefId;

#[derive(Clone, Default)]
pub struct SelectionCache<'tcx> {
    pub hashmap: Lock<
        FxHashMap<
            ty::ParamEnvAnd<'tcx, ty::TraitRef<'tcx>>,
            WithDepNode<SelectionResult<'tcx, SelectionCandidate<'tcx>>>,
        >,
    >,
}

impl<'tcx> SelectionCache<'tcx> {
    /// Actually frees the underlying memory in contrast to what stdlib containers do on `clear`
    pub fn clear(&self) {
        *self.hashmap.borrow_mut() = Default::default();
    }
}

/// The selection process begins by considering all impls, where
/// clauses, and so forth that might resolve an obligation. Sometimes
/// we'll be able to say definitively that (e.g.) an impl does not
/// apply to the obligation: perhaps it is defined for `usize` but the
/// obligation is for `int`. In that case, we drop the impl out of the
/// list. But the other cases are considered *candidates*.
///
/// For selection to succeed, there must be exactly one matching
/// candidate. If the obligation is fully known, this is guaranteed
/// by coherence. However, if the obligation contains type parameters
/// or variables, there may be multiple such impls.
///
/// It is not a real problem if multiple matching impls exist because
/// of type variables - it just means the obligation isn't sufficiently
/// elaborated. In that case we report an ambiguity, and the caller can
/// try again after more type information has been gathered or report a
/// "type annotations needed" error.
///
/// However, with type parameters, this can be a real problem - type
/// parameters don't unify with regular types, but they *can* unify
/// with variables from blanket impls, and (unless we know its bounds
/// will always be satisfied) picking the blanket impl will be wrong
/// for at least *some* substitutions. To make this concrete, if we have
///
///    trait AsDebug { type Out : fmt::Debug; fn debug(self) -> Self::Out; }
///    impl<T: fmt::Debug> AsDebug for T {
///        type Out = T;
///        fn debug(self) -> fmt::Debug { self }
///    }
///    fn foo<T: AsDebug>(t: T) { println!("{:?}", <T as AsDebug>::debug(t)); }
///
/// we can't just use the impl to resolve the `<T as AsDebug>` obligation
/// -- a type from another crate (that doesn't implement `fmt::Debug`) could
/// implement `AsDebug`.
///
/// Because where-clauses match the type exactly, multiple clauses can
/// only match if there are unresolved variables, and we can mostly just
/// report this ambiguity in that case. This is still a problem - we can't
/// *do anything* with ambiguities that involve only regions. This is issue
/// #21974.
///
/// If a single where-clause matches and there are no inference
/// variables left, then it definitely matches and we can just select
/// it.
///
/// In fact, we even select the where-clause when the obligation contains
/// inference variables. The can lead to inference making "leaps of logic",
/// for example in this situation:
///
///    pub trait Foo<T> { fn foo(&self) -> T; }
///    impl<T> Foo<()> for T { fn foo(&self) { } }
///    impl Foo<bool> for bool { fn foo(&self) -> bool { *self } }
///
///    pub fn foo<T>(t: T) where T: Foo<bool> {
///       println!("{:?}", <T as Foo<_>>::foo(&t));
///    }
///    fn main() { foo(false); }
///
/// Here the obligation `<T as Foo<$0>>` can be matched by both the blanket
/// impl and the where-clause. We select the where-clause and unify `$0=bool`,
/// so the program prints "false". However, if the where-clause is omitted,
/// the blanket impl is selected, we unify `$0=()`, and the program prints
/// "()".
///
/// Exactly the same issues apply to projection and object candidates, except
/// that we can have both a projection candidate and a where-clause candidate
/// for the same obligation. In that case either would do (except that
/// different "leaps of logic" would occur if inference variables are
/// present), and we just pick the where-clause. This is, for example,
/// required for associated types to work in default impls, as the bounds
/// are visible both as projection bounds and as where-clauses from the
/// parameter environment.
#[derive(PartialEq, Eq, Debug, Clone, TypeFoldable)]
pub enum SelectionCandidate<'tcx> {
    BuiltinCandidate {
        /// `false` if there are no *further* obligations.
        has_nested: bool,
    },
    ParamCandidate(ty::PolyTraitRef<'tcx>),
    ImplCandidate(DefId),
    AutoImplCandidate(DefId),

    /// This is a trait matching with a projected type as `Self`, and
    /// we found an applicable bound in the trait definition.
    ProjectionCandidate,

    /// Implementation of a `Fn`-family trait by one of the anonymous types
    /// generated for a `||` expression.
    ClosureCandidate,

    /// Implementation of a `Generator` trait by one of the anonymous types
    /// generated for a generator.
    GeneratorCandidate,

    /// Implementation of a `Fn`-family trait by one of the anonymous
    /// types generated for a fn pointer type (e.g., `fn(int) -> int`)
    FnPointerCandidate,

    TraitAliasCandidate(DefId),

    ObjectCandidate,

    BuiltinObjectCandidate,

    BuiltinUnsizeCandidate,
}

/// The result of trait evaluation. The order is important
/// here as the evaluation of a list is the maximum of the
/// evaluations.
///
/// The evaluation results are ordered:
///     - `EvaluatedToOk` implies `EvaluatedToOkModuloRegions`
///       implies `EvaluatedToAmbig` implies `EvaluatedToUnknown`
///     - `EvaluatedToErr` implies `EvaluatedToRecur`
///     - the "union" of evaluation results is equal to their maximum -
///     all the "potential success" candidates can potentially succeed,
///     so they are noops when unioned with a definite error, and within
///     the categories it's easy to see that the unions are correct.
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, HashStable)]
pub enum EvaluationResult {
    /// Evaluation successful.
    EvaluatedToOk,
    /// Evaluation successful, but there were unevaluated region obligations.
    EvaluatedToOkModuloRegions,
    /// Evaluation is known to be ambiguous -- it *might* hold for some
    /// assignment of inference variables, but it might not.
    ///
    /// While this has the same meaning as `EvaluatedToUnknown` -- we can't
    /// know whether this obligation holds or not -- it is the result we
    /// would get with an empty stack, and therefore is cacheable.
    EvaluatedToAmbig,
    /// Evaluation failed because of recursion involving inference
    /// variables. We are somewhat imprecise there, so we don't actually
    /// know the real result.
    ///
    /// This can't be trivially cached for the same reason as `EvaluatedToRecur`.
    EvaluatedToUnknown,
    /// Evaluation failed because we encountered an obligation we are already
    /// trying to prove on this branch.
    ///
    /// We know this branch can't be a part of a minimal proof-tree for
    /// the "root" of our cycle, because then we could cut out the recursion
    /// and maintain a valid proof tree. However, this does not mean
    /// that all the obligations on this branch do not hold -- it's possible
    /// that we entered this branch "speculatively", and that there
    /// might be some other way to prove this obligation that does not
    /// go through this cycle -- so we can't cache this as a failure.
    ///
    /// For example, suppose we have this:
    ///
    /// ```rust,ignore (pseudo-Rust)
    /// pub trait Trait { fn xyz(); }
    /// // This impl is "useless", but we can still have
    /// // an `impl Trait for SomeUnsizedType` somewhere.
    /// impl<T: Trait + Sized> Trait for T { fn xyz() {} }
    ///
    /// pub fn foo<T: Trait + ?Sized>() {
    ///     <T as Trait>::xyz();
    /// }
    /// ```
    ///
    /// When checking `foo`, we have to prove `T: Trait`. This basically
    /// translates into this:
    ///
    /// ```plain,ignore
    /// (T: Trait + Sized →_\impl T: Trait), T: Trait ⊢ T: Trait
    /// ```
    ///
    /// When we try to prove it, we first go the first option, which
    /// recurses. This shows us that the impl is "useless" -- it won't
    /// tell us that `T: Trait` unless it already implemented `Trait`
    /// by some other means. However, that does not prevent `T: Trait`
    /// does not hold, because of the bound (which can indeed be satisfied
    /// by `SomeUnsizedType` from another crate).
    //
    // FIXME: when an `EvaluatedToRecur` goes past its parent root, we
    // ought to convert it to an `EvaluatedToErr`, because we know
    // there definitely isn't a proof tree for that obligation. Not
    // doing so is still sound -- there isn't any proof tree, so the
    // branch still can't be a part of a minimal one -- but does not re-enable caching.
    EvaluatedToRecur,
    /// Evaluation failed.
    EvaluatedToErr,
}

impl EvaluationResult {
    /// Returns `true` if this evaluation result is known to apply, even
    /// considering outlives constraints.
    pub fn must_apply_considering_regions(self) -> bool {
        self == EvaluatedToOk
    }

    /// Returns `true` if this evaluation result is known to apply, ignoring
    /// outlives constraints.
    pub fn must_apply_modulo_regions(self) -> bool {
        self <= EvaluatedToOkModuloRegions
    }

    pub fn may_apply(self) -> bool {
        match self {
            EvaluatedToOk | EvaluatedToOkModuloRegions | EvaluatedToAmbig | EvaluatedToUnknown => {
                true
            }

            EvaluatedToErr | EvaluatedToRecur => false,
        }
    }

    pub fn is_stack_dependent(self) -> bool {
        match self {
            EvaluatedToUnknown | EvaluatedToRecur => true,

            EvaluatedToOk | EvaluatedToOkModuloRegions | EvaluatedToAmbig | EvaluatedToErr => false,
        }
    }
}

/// Indicates that trait evaluation caused overflow.
#[derive(Copy, Clone, Debug, PartialEq, Eq, HashStable)]
pub struct OverflowError;

impl<'tcx> From<OverflowError> for SelectionError<'tcx> {
    fn from(OverflowError: OverflowError) -> SelectionError<'tcx> {
        SelectionError::Overflow
    }
}

#[derive(Clone, Default)]
pub struct EvaluationCache<'tcx> {
    pub hashmap: Lock<
        FxHashMap<ty::ParamEnvAnd<'tcx, ty::PolyTraitRef<'tcx>>, WithDepNode<EvaluationResult>>,
    >,
}

impl<'tcx> EvaluationCache<'tcx> {
    /// Actually frees the underlying memory in contrast to what stdlib containers do on `clear`
    pub fn clear(&self) {
        *self.hashmap.borrow_mut() = Default::default();
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct WithDepNode<T> {
    dep_node: DepNodeIndex,
    cached_value: T,
}

impl<T: Clone> WithDepNode<T> {
    pub fn new(dep_node: DepNodeIndex, cached_value: T) -> Self {
        WithDepNode { dep_node, cached_value }
    }

    pub fn get(&self, tcx: TyCtxt<'_>) -> T {
        tcx.dep_graph.read_index(self.dep_node);
        self.cached_value.clone()
    }
}

#[derive(Clone, Debug)]
pub enum IntercrateAmbiguityCause {
    DownstreamCrate { trait_desc: String, self_desc: Option<String> },
    UpstreamCrateUpdate { trait_desc: String, self_desc: Option<String> },
    ReservationImpl { message: String },
}

impl IntercrateAmbiguityCause {
    /// Emits notes when the overlap is caused by complex intercrate ambiguities.
    /// See #23980 for details.
    pub fn add_intercrate_ambiguity_hint(&self, err: &mut rustc_errors::DiagnosticBuilder<'_>) {
        err.note(&self.intercrate_ambiguity_hint());
    }

    pub fn intercrate_ambiguity_hint(&self) -> String {
        match self {
            &IntercrateAmbiguityCause::DownstreamCrate { ref trait_desc, ref self_desc } => {
                let self_desc = if let &Some(ref ty) = self_desc {
                    format!(" for type `{}`", ty)
                } else {
                    String::new()
                };
                format!("downstream crates may implement trait `{}`{}", trait_desc, self_desc)
            }
            &IntercrateAmbiguityCause::UpstreamCrateUpdate { ref trait_desc, ref self_desc } => {
                let self_desc = if let &Some(ref ty) = self_desc {
                    format!(" for type `{}`", ty)
                } else {
                    String::new()
                };
                format!(
                    "upstream crates may add a new impl of trait `{}`{} \
                     in future versions",
                    trait_desc, self_desc
                )
            }
            &IntercrateAmbiguityCause::ReservationImpl { ref message } => message.clone(),
        }
    }
}
