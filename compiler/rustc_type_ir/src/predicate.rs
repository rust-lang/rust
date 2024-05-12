use std::fmt;

#[cfg(feature = "nightly")]
use rustc_macros::{Decodable, Encodable, HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::inherent::*;
use crate::visit::TypeVisitableExt as _;
use crate::{DebugWithInfcx, Interner};

/// A complete reference to a trait. These take numerous guises in syntax,
/// but perhaps the most recognizable form is in a where-clause:
/// ```ignore (illustrative)
/// T: Foo<U>
/// ```
/// This would be represented by a trait-reference where the `DefId` is the
/// `DefId` for the trait `Foo` and the args define `T` as parameter 0,
/// and `U` as parameter 1.
///
/// Trait references also appear in object types like `Foo<U>`, but in
/// that case the `Self` parameter is absent from the generic parameters.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct TraitRef<I: Interner> {
    pub def_id: I::DefId,
    pub args: I::GenericArgs,
    /// This field exists to prevent the creation of `TraitRef` without
    /// calling [`TraitRef::new`].
    _use_trait_ref_new_instead: (),
}

impl<I: Interner> TraitRef<I> {
    pub fn new(
        interner: I,
        trait_def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self {
        let args = interner.check_and_mk_args(trait_def_id, args);
        Self { def_id: trait_def_id, args, _use_trait_ref_new_instead: () }
    }

    pub fn from_method(interner: I, trait_id: I::DefId, args: I::GenericArgs) -> TraitRef<I> {
        let generics = interner.generics_of(trait_id);
        TraitRef::new(interner, trait_id, args.into_iter().take(generics.count()))
    }

    /// Returns a `TraitRef` of the form `P0: Foo<P1..Pn>` where `Pi`
    /// are the parameters defined on trait.
    pub fn identity(interner: I, def_id: I::DefId) -> TraitRef<I> {
        TraitRef::new(interner, def_id, I::GenericArgs::identity_for_item(interner, def_id))
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        TraitRef::new(
            interner,
            self.def_id,
            [self_ty.into()].into_iter().chain(self.args.into_iter().skip(1)),
        )
    }

    #[inline]
    pub fn self_ty(&self) -> I::Ty {
        self.args.type_at(0)
    }
}

#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct TraitPredicate<I: Interner> {
    pub trait_ref: TraitRef<I>,

    /// If polarity is Positive: we are proving that the trait is implemented.
    ///
    /// If polarity is Negative: we are proving that a negative impl of this trait
    /// exists. (Note that coherence also checks whether negative impls of supertraits
    /// exist via a series of predicates.)
    ///
    /// If polarity is Reserved: that's a bug.
    pub polarity: PredicatePolarity,
}

impl<I: Interner> TraitPredicate<I> {
    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        Self { trait_ref: self.trait_ref.with_self_ty(interner, self_ty), polarity: self.polarity }
    }

    pub fn def_id(self) -> I::DefId {
        self.trait_ref.def_id
    }

    pub fn self_ty(self) -> I::Ty {
        self.trait_ref.self_ty()
    }
}

impl<I: Interner> fmt::Debug for TraitPredicate<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME(effects) printing?
        write!(f, "TraitPredicate({:?}, polarity:{:?})", self.trait_ref, self.polarity)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative,
    /// `#[rustc_reservation_impl] impl Trait for Type`
    ///
    /// This is a "stability hack", not a real Rust feature.
    /// See #64631 for details.
    Reservation,
}

impl fmt::Display for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positive => f.write_str("positive"),
            Self::Negative => f.write_str("negative"),
            Self::Reservation => f.write_str("reservation"),
        }
    }
}

/// Polarity for a trait predicate. May either be negative or positive.
/// Distinguished from [`ImplPolarity`] since we never compute goals with
/// "reservation" level.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub enum PredicatePolarity {
    /// `Type: Trait`
    Positive,
    /// `Type: !Trait`
    Negative,
}

impl PredicatePolarity {
    /// Flips polarity by turning `Positive` into `Negative` and `Negative` into `Positive`.
    pub fn flip(&self) -> PredicatePolarity {
        match self {
            PredicatePolarity::Positive => PredicatePolarity::Negative,
            PredicatePolarity::Negative => PredicatePolarity::Positive,
        }
    }
}

impl fmt::Display for PredicatePolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positive => f.write_str("positive"),
            Self::Negative => f.write_str("negative"),
        }
    }
}

#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = ""),
    Debug(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub enum ExistentialPredicate<I: Interner> {
    /// E.g., `Iterator`.
    Trait(ExistentialTraitRef<I>),
    /// E.g., `Iterator::Item = T`.
    Projection(ExistentialProjection<I>),
    /// E.g., `Send`.
    AutoTrait(I::DefId),
}

// FIXME: Implement this the right way after
impl<I: Interner> DebugWithInfcx<I> for ExistentialPredicate<I> {
    fn fmt<Infcx: rustc_type_ir::InferCtxtLike<Interner = I>>(
        this: rustc_type_ir::WithInfcx<'_, Infcx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Debug::fmt(&this.data, f)
    }
}

/// An existential reference to a trait, where `Self` is erased.
/// For example, the trait object `Trait<'a, 'b, X, Y>` is:
/// ```ignore (illustrative)
/// exists T. T: Trait<'a, 'b, X, Y>
/// ```
/// The generic parameters don't include the erased `Self`, only trait
/// type and lifetime parameters (`[X, Y]` and `['a, 'b]` above).
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct ExistentialTraitRef<I: Interner> {
    pub def_id: I::DefId,
    pub args: I::GenericArgs,
}

impl<I: Interner> ExistentialTraitRef<I> {
    pub fn erase_self_ty(interner: I, trait_ref: TraitRef<I>) -> ExistentialTraitRef<I> {
        // Assert there is a Self.
        trait_ref.args.type_at(0);

        ExistentialTraitRef {
            def_id: trait_ref.def_id,
            args: interner.mk_args(&trait_ref.args[1..]),
        }
    }

    /// Object types don't have a self type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self type. A common choice is `mk_err()`
    /// or some placeholder type.
    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> TraitRef<I> {
        // otherwise the escaping vars would be captured by the binder
        // debug_assert!(!self_ty.has_escaping_bound_vars());

        TraitRef::new(
            interner,
            self.def_id,
            [self_ty.into()].into_iter().chain(self.args.into_iter()),
        )
    }
}

/// A `ProjectionPredicate` for an `ExistentialTraitRef`.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct ExistentialProjection<I: Interner> {
    pub def_id: I::DefId,
    pub args: I::GenericArgs,
    pub term: I::Term,
}

impl<I: Interner> ExistentialProjection<I> {
    /// Extracts the underlying existential trait reference from this projection.
    /// For example, if this is a projection of `exists T. <T as Iterator>::Item == X`,
    /// then this function would return an `exists T. T: Iterator` existential trait
    /// reference.
    pub fn trait_ref(&self, tcx: I) -> ExistentialTraitRef<I> {
        let def_id = tcx.parent(self.def_id);
        let args_count = tcx.generics_of(def_id).count() - 1;
        let args = tcx.mk_args(&self.args[..args_count]);
        ExistentialTraitRef { def_id, args }
    }

    pub fn with_self_ty(&self, tcx: I, self_ty: I::Ty) -> ProjectionPredicate<I> {
        // otherwise the escaping regions would be captured by the binders
        debug_assert!(!self_ty.has_escaping_bound_vars());

        ProjectionPredicate {
            projection_term: I::AliasTerm::new(
                tcx,
                self.def_id,
                [self_ty.into()].into_iter().chain(self.args),
            ),
            term: self.term,
        }
    }

    pub fn erase_self_ty(tcx: I, projection_predicate: ProjectionPredicate<I>) -> Self {
        // Assert there is a Self.
        projection_predicate.projection_term.args().type_at(0);

        Self {
            def_id: projection_predicate.projection_term.def_id(),
            args: tcx.mk_args(&projection_predicate.projection_term.args()[1..]),
            term: projection_predicate.term,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_NoContext))]
pub enum AliasTermKind {
    /// A projection `<Type as Trait>::AssocType`.
    /// Can get normalized away if monomorphic enough.
    ProjectionTy,
    /// An associated type in an inherent `impl`
    InherentTy,
    /// An opaque type (usually from `impl Trait` in type aliases or function return types)
    /// Can only be normalized away in RevealAll mode
    OpaqueTy,
    /// A type alias that actually checks its trait bounds.
    /// Currently only used if the type alias references opaque types.
    /// Can always be normalized away.
    WeakTy,
    /// An unevaluated const coming from a generic const expression.
    UnevaluatedConst,
    /// An unevaluated const coming from an associated const.
    ProjectionConst,
}

impl AliasTermKind {
    pub fn descr(self) -> &'static str {
        match self {
            AliasTermKind::ProjectionTy => "associated type",
            AliasTermKind::ProjectionConst => "associated const",
            AliasTermKind::InherentTy => "inherent associated type",
            AliasTermKind::OpaqueTy => "opaque type",
            AliasTermKind::WeakTy => "type alias",
            AliasTermKind::UnevaluatedConst => "unevaluated constant",
        }
    }
}

/// This kind of predicate has no *direct* correspondent in the
/// syntax, but it roughly corresponds to the syntactic forms:
///
/// 1. `T: TraitRef<..., Item = Type>`
/// 2. `<T as TraitRef<...>>::Item == Type` (NYI)
///
/// In particular, form #1 is "desugared" to the combination of a
/// normal trait predicate (`T: TraitRef<...>`) and one of these
/// predicates. Form #2 is a broader form in that it also permits
/// equality between arbitrary types. Processing an instance of
/// Form #2 eventually yields one of these `ProjectionPredicate`
/// instances to normalize the LHS.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct ProjectionPredicate<I: Interner> {
    pub projection_term: I::AliasTerm,
    pub term: I::Term,
}

impl<I: Interner> ProjectionPredicate<I> {
    pub fn self_ty(self) -> I::Ty {
        self.projection_term.self_ty()
    }

    pub fn with_self_ty(self, tcx: I, self_ty: I::Ty) -> ProjectionPredicate<I> {
        Self { projection_term: self.projection_term.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn trait_def_id(self, tcx: I) -> I::DefId {
        self.projection_term.trait_def_id(tcx)
    }

    pub fn def_id(self) -> I::DefId {
        self.projection_term.def_id()
    }
}

impl<I: Interner> fmt::Debug for ProjectionPredicate<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProjectionPredicate({:?}, {:?})", self.projection_term, self.term)
    }
}

/// Used by the new solver. Unlike a `ProjectionPredicate` this can only be
/// proven by actually normalizing `alias`.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct NormalizesTo<I: Interner> {
    pub alias: I::AliasTerm,
    pub term: I::Term,
}

impl<I: Interner> NormalizesTo<I> {
    pub fn self_ty(self) -> I::Ty {
        self.alias.self_ty()
    }

    pub fn with_self_ty(self, tcx: I, self_ty: I::Ty) -> NormalizesTo<I> {
        Self { alias: self.alias.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn trait_def_id(self, tcx: I) -> I::DefId {
        self.alias.trait_def_id(tcx)
    }

    pub fn def_id(self) -> I::DefId {
        self.alias.def_id()
    }
}

impl<I: Interner> fmt::Debug for NormalizesTo<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NormalizesTo({:?}, {:?})", self.alias, self.term)
    }
}

/// Encodes that `a` must be a subtype of `b`. The `a_is_expected` flag indicates
/// whether the `a` type is the type that we should label as "expected" when
/// presenting user diagnostics.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = ""),
    Debug(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct SubtypePredicate<I: Interner> {
    pub a_is_expected: bool,
    pub a: I::Ty,
    pub b: I::Ty,
}

/// Encodes that we have to coerce *from* the `a` type to the `b` type.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Hash(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = ""),
    Debug(bound = "")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct CoercePredicate<I: Interner> {
    pub a: I::Ty,
    pub b: I::Ty,
}
