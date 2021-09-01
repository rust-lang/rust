//! Candidate selection. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html#selection

use self::EvaluationResult::*;

use super::{SelectionError, SelectionResult};

use crate::ty;

use rustc_hir::def_id::DefId;
use rustc_query_system::cache::Cache;

pub type SelectionCache<'tcx> = Cache<
    ty::ConstnessAnd<ty::ParamEnvAnd<'tcx, ty::TraitRef<'tcx>>>,
    SelectionResult<'tcx, SelectionCandidate<'tcx>>,
>;

pub type EvaluationCache<'tcx> =
    Cache<ty::ParamEnvAnd<'tcx, ty::ConstnessAnd<ty::PolyTraitRef<'tcx>>>, EvaluationResult>;

/// The selection process begins by considering all impls, where
/// clauses, and so forth that might resolve an obligation. Sometimes
/// we'll be able to say definitively that (e.g.) an impl does not
/// apply to the obligation: perhaps it is defined for `usize` but the
/// obligation is for `i32`. In that case, we drop the impl out of the
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
/// ```rust, ignore
/// trait AsDebug { type Out: fmt::Debug; fn debug(self) -> Self::Out; }
/// impl<T: fmt::Debug> AsDebug for T {
///     type Out = T;
///     fn debug(self) -> fmt::Debug { self }
/// }
/// fn foo<T: AsDebug>(t: T) { println!("{:?}", <T as AsDebug>::debug(t)); }
/// ```
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
/// ```rust, ignore
/// pub trait Foo<T> { fn foo(&self) -> T; }
/// impl<T> Foo<()> for T { fn foo(&self) { } }
/// impl Foo<bool> for bool { fn foo(&self) -> bool { *self } }
///
/// pub fn foo<T>(t: T) where T: Foo<bool> {
///     println!("{:?}", <T as Foo<_>>::foo(&t));
/// }
/// fn main() { foo(false); }
/// ```
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
    ParamCandidate(ty::ConstnessAnd<ty::PolyTraitRef<'tcx>>),
    ImplCandidate(DefId),
    AutoImplCandidate(DefId),

    /// This is a trait matching with a projected type as `Self`, and we found
    /// an applicable bound in the trait definition. The `usize` is an index
    /// into the list returned by `tcx.item_bounds`.
    ProjectionCandidate(usize),

    /// Implementation of a `Fn`-family trait by one of the anonymous types
    /// generated for an `||` expression.
    ClosureCandidate,

    /// Implementation of a `Generator` trait by one of the anonymous types
    /// generated for a generator.
    GeneratorCandidate,

    /// Implementation of a `Fn`-family trait by one of the anonymous
    /// types generated for a fn pointer type (e.g., `fn(int) -> int`)
    FnPointerCandidate,

    /// Builtin implementation of `DiscriminantKind`.
    DiscriminantKindCandidate,

    /// Builtin implementation of `Pointee`.
    PointeeCandidate,

    TraitAliasCandidate(DefId),

    /// Matching `dyn Trait` with a supertrait of `Trait`. The index is the
    /// position in the iterator returned by
    /// `rustc_infer::traits::util::supertraits`.
    ObjectCandidate(usize),

    /// Perform trait upcasting coercion of `dyn Trait` to a supertrait of `Trait`.
    /// The index is the position in the iterator returned by
    /// `rustc_infer::traits::util::supertraits`.
    TraitUpcastingUnsizeCandidate(usize),

    BuiltinObjectCandidate,

    BuiltinUnsizeCandidate,

    /// Implementation of `const Drop`.
    ConstDropCandidate,
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
