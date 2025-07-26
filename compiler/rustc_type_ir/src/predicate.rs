use std::fmt;
use std::hash::Hash;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{
    Decodable, Decodable_NoContext, Encodable, Encodable_NoContext, HashStable_NoContext,
};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::inherent::*;
use crate::lift::Lift;
use crate::upcast::{Upcast, UpcastFrom};
use crate::visit::TypeVisitableExt as _;
use crate::{self as ty, Interner};

/// `A: 'region`
#[derive_where(Clone; I: Interner, A: Clone)]
#[derive_where(Copy; I: Interner, A: Copy)]
#[derive_where(Hash; I: Interner, A: Hash)]
#[derive_where(PartialEq; I: Interner, A: PartialEq)]
#[derive_where(Eq; I: Interner, A: Eq)]
#[derive_where(Debug; I: Interner, A: fmt::Debug)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct OutlivesPredicate<I: Interner, A>(pub A, pub I::Region);

// FIXME: We manually derive `Lift` because the `derive(Lift_Generic)` doesn't
// understand how to turn `A` to `A::Lifted` in the output `type Lifted`.
impl<I: Interner, U: Interner, A> Lift<U> for OutlivesPredicate<I, A>
where
    A: Lift<U>,
    I::Region: Lift<U, Lifted = U::Region>,
{
    type Lifted = OutlivesPredicate<U, A::Lifted>;

    fn lift_to_interner(self, cx: U) -> Option<Self::Lifted> {
        Some(OutlivesPredicate(self.0.lift_to_interner(cx)?, self.1.lift_to_interner(cx)?))
    }
}

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
#[derive_where(Clone, Copy, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct TraitRef<I: Interner> {
    pub def_id: I::DefId,
    pub args: I::GenericArgs,
    /// This field exists to prevent the creation of `TraitRef` without
    /// calling [`TraitRef::new_from_args`].
    _use_trait_ref_new_instead: (),
}

impl<I: Interner> TraitRef<I> {
    pub fn new_from_args(interner: I, trait_def_id: I::DefId, args: I::GenericArgs) -> Self {
        interner.debug_assert_args_compatible(trait_def_id, args);
        Self { def_id: trait_def_id, args, _use_trait_ref_new_instead: () }
    }

    pub fn new(
        interner: I,
        trait_def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self {
        let args = interner.mk_args_from_iter(args.into_iter().map(Into::into));
        Self::new_from_args(interner, trait_def_id, args)
    }

    pub fn from_method(interner: I, trait_id: I::DefId, args: I::GenericArgs) -> TraitRef<I> {
        let generics = interner.generics_of(trait_id);
        TraitRef::new(interner, trait_id, args.iter().take(generics.count()))
    }

    /// Returns a `TraitRef` of the form `P0: Foo<P1..Pn>` where `Pi`
    /// are the parameters defined on trait.
    pub fn identity(interner: I, def_id: I::DefId) -> TraitRef<I> {
        TraitRef::new_from_args(
            interner,
            def_id,
            I::GenericArgs::identity_for_item(interner, def_id),
        )
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        TraitRef::new(
            interner,
            self.def_id,
            [self_ty.into()].into_iter().chain(self.args.iter().skip(1)),
        )
    }

    #[inline]
    pub fn self_ty(&self) -> I::Ty {
        self.args.type_at(0)
    }
}

impl<I: Interner> ty::Binder<I, TraitRef<I>> {
    pub fn self_ty(&self) -> ty::Binder<I, I::Ty> {
        self.map_bound_ref(|tr| tr.self_ty())
    }

    pub fn def_id(&self) -> I::DefId {
        self.skip_binder().def_id
    }

    pub fn to_host_effect_clause(self, cx: I, constness: BoundConstness) -> I::Clause {
        self.map_bound(|trait_ref| {
            ty::ClauseKind::HostEffect(HostEffectPredicate { trait_ref, constness })
        })
        .upcast(cx)
    }
}

#[derive_where(Clone, Copy, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct TraitPredicate<I: Interner> {
    pub trait_ref: TraitRef<I>,

    /// If polarity is Positive: we are proving that the trait is implemented.
    ///
    /// If polarity is Negative: we are proving that a negative impl of this trait
    /// exists. (Note that coherence also checks whether negative impls of supertraits
    /// exist via a series of predicates.)
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

impl<I: Interner> ty::Binder<I, TraitPredicate<I>> {
    pub fn def_id(self) -> I::DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().def_id()
    }

    pub fn self_ty(self) -> ty::Binder<I, I::Ty> {
        self.map_bound(|trait_ref| trait_ref.self_ty())
    }

    #[inline]
    pub fn polarity(self) -> PredicatePolarity {
        self.skip_binder().polarity
    }
}

impl<I: Interner> UpcastFrom<I, TraitRef<I>> for TraitPredicate<I> {
    fn upcast_from(from: TraitRef<I>, _tcx: I) -> Self {
        TraitPredicate { trait_ref: from, polarity: PredicatePolarity::Positive }
    }
}

impl<I: Interner> fmt::Debug for TraitPredicate<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TraitPredicate({:?}, polarity:{:?})", self.trait_ref, self.polarity)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
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

impl ImplPolarity {
    /// The polarity marker in front of the impl trait ref if applicable.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Positive => "",
            Self::Negative => "!",
            Self::Reservation => "",
        }
    }
}

/// Polarity for a trait predicate. May either be negative or positive.
/// Distinguished from [`ImplPolarity`] since we never compute goals with
/// "reservation" level.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
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

#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub enum ExistentialPredicate<I: Interner> {
    /// E.g., `Iterator`.
    Trait(ExistentialTraitRef<I>),
    /// E.g., `Iterator::Item = T`.
    Projection(ExistentialProjection<I>),
    /// E.g., `Send`.
    AutoTrait(I::DefId),
}

impl<I: Interner> ty::Binder<I, ExistentialPredicate<I>> {
    /// Given an existential predicate like `?Self: PartialEq<u32>` (e.g., derived from `dyn PartialEq<u32>`),
    /// and a concrete type `self_ty`, returns a full predicate where the existentially quantified variable `?Self`
    /// has been replaced with `self_ty` (e.g., `self_ty: PartialEq<u32>`, in our example).
    pub fn with_self_ty(&self, cx: I, self_ty: I::Ty) -> I::Clause {
        match self.skip_binder() {
            ExistentialPredicate::Trait(tr) => self.rebind(tr).with_self_ty(cx, self_ty).upcast(cx),
            ExistentialPredicate::Projection(p) => {
                self.rebind(p.with_self_ty(cx, self_ty)).upcast(cx)
            }
            ExistentialPredicate::AutoTrait(did) => {
                let generics = cx.generics_of(did);
                let trait_ref = if generics.count() == 1 {
                    ty::TraitRef::new(cx, did, [self_ty])
                } else {
                    // If this is an ill-formed auto trait, then synthesize
                    // new error args for the missing generics.
                    let err_args = GenericArgs::extend_with_error(cx, did, &[self_ty.into()]);
                    ty::TraitRef::new_from_args(cx, did, err_args)
                };
                self.rebind(trait_ref).upcast(cx)
            }
        }
    }
}

/// An existential reference to a trait, where `Self` is erased.
/// For example, the trait object `Trait<'a, 'b, X, Y>` is:
/// ```ignore (illustrative)
/// exists T. T: Trait<'a, 'b, X, Y>
/// ```
/// The generic parameters don't include the erased `Self`, only trait
/// type and lifetime parameters (`[X, Y]` and `['a, 'b]` above).
#[derive_where(Clone, Copy, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct ExistentialTraitRef<I: Interner> {
    pub def_id: I::DefId,
    pub args: I::GenericArgs,
    /// This field exists to prevent the creation of `ExistentialTraitRef` without
    /// calling [`ExistentialTraitRef::new_from_args`].
    _use_existential_trait_ref_new_instead: (),
}

impl<I: Interner> ExistentialTraitRef<I> {
    pub fn new_from_args(interner: I, trait_def_id: I::DefId, args: I::GenericArgs) -> Self {
        interner.debug_assert_existential_args_compatible(trait_def_id, args);
        Self { def_id: trait_def_id, args, _use_existential_trait_ref_new_instead: () }
    }

    pub fn new(
        interner: I,
        trait_def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self {
        let args = interner.mk_args_from_iter(args.into_iter().map(Into::into));
        Self::new_from_args(interner, trait_def_id, args)
    }

    pub fn erase_self_ty(interner: I, trait_ref: TraitRef<I>) -> ExistentialTraitRef<I> {
        // Assert there is a Self.
        trait_ref.args.type_at(0);

        ExistentialTraitRef {
            def_id: trait_ref.def_id,
            args: interner.mk_args(&trait_ref.args.as_slice()[1..]),
            _use_existential_trait_ref_new_instead: (),
        }
    }

    /// Object types don't have a self type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self type. A common choice is `mk_err()`
    /// or some placeholder type.
    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> TraitRef<I> {
        // otherwise the escaping vars would be captured by the binder
        // debug_assert!(!self_ty.has_escaping_bound_vars());

        TraitRef::new(interner, self.def_id, [self_ty.into()].into_iter().chain(self.args.iter()))
    }
}

impl<I: Interner> ty::Binder<I, ExistentialTraitRef<I>> {
    pub fn def_id(&self) -> I::DefId {
        self.skip_binder().def_id
    }

    /// Object types don't have a self type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self type. A common choice is `mk_err()`
    /// or some placeholder type.
    pub fn with_self_ty(&self, cx: I, self_ty: I::Ty) -> ty::Binder<I, TraitRef<I>> {
        self.map_bound(|trait_ref| trait_ref.with_self_ty(cx, self_ty))
    }
}

/// A `ProjectionPredicate` for an `ExistentialTraitRef`.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct ExistentialProjection<I: Interner> {
    pub def_id: I::DefId,
    pub args: I::GenericArgs,
    pub term: I::Term,

    /// This field exists to prevent the creation of `ExistentialProjection`
    /// without using [`ExistentialProjection::new_from_args`].
    use_existential_projection_new_instead: (),
}

impl<I: Interner> ExistentialProjection<I> {
    pub fn new_from_args(
        interner: I,
        def_id: I::DefId,
        args: I::GenericArgs,
        term: I::Term,
    ) -> ExistentialProjection<I> {
        interner.debug_assert_existential_args_compatible(def_id, args);
        Self { def_id, args, term, use_existential_projection_new_instead: () }
    }

    pub fn new(
        interner: I,
        def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
        term: I::Term,
    ) -> ExistentialProjection<I> {
        let args = interner.mk_args_from_iter(args.into_iter().map(Into::into));
        Self::new_from_args(interner, def_id, args, term)
    }

    /// Extracts the underlying existential trait reference from this projection.
    /// For example, if this is a projection of `exists T. <T as Iterator>::Item == X`,
    /// then this function would return an `exists T. T: Iterator` existential trait
    /// reference.
    pub fn trait_ref(&self, interner: I) -> ExistentialTraitRef<I> {
        let def_id = interner.parent(self.def_id);
        let args_count = interner.generics_of(def_id).count() - 1;
        let args = interner.mk_args(&self.args.as_slice()[..args_count]);
        ExistentialTraitRef { def_id, args, _use_existential_trait_ref_new_instead: () }
    }

    pub fn with_self_ty(&self, interner: I, self_ty: I::Ty) -> ProjectionPredicate<I> {
        // otherwise the escaping regions would be captured by the binders
        debug_assert!(!self_ty.has_escaping_bound_vars());

        ProjectionPredicate {
            projection_term: AliasTerm::new(
                interner,
                self.def_id,
                [self_ty.into()].iter().chain(self.args.iter()),
            ),
            term: self.term,
        }
    }

    pub fn erase_self_ty(interner: I, projection_predicate: ProjectionPredicate<I>) -> Self {
        // Assert there is a Self.
        projection_predicate.projection_term.args.type_at(0);

        Self {
            def_id: projection_predicate.projection_term.def_id,
            args: interner.mk_args(&projection_predicate.projection_term.args.as_slice()[1..]),
            term: projection_predicate.term,
            use_existential_projection_new_instead: (),
        }
    }
}

impl<I: Interner> ty::Binder<I, ExistentialProjection<I>> {
    pub fn with_self_ty(&self, cx: I, self_ty: I::Ty) -> ty::Binder<I, ProjectionPredicate<I>> {
        self.map_bound(|p| p.with_self_ty(cx, self_ty))
    }

    pub fn item_def_id(&self) -> I::DefId {
        self.skip_binder().def_id
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
    /// Can only be normalized away in PostAnalysis mode or its defining scope.
    OpaqueTy,
    /// A free type alias that actually checks its trait bounds.
    /// Currently only used if the type alias references opaque types.
    /// Can always be normalized away.
    FreeTy,

    /// An unevaluated anonymous constants.
    UnevaluatedConst,
    /// An unevaluated const coming from an associated const.
    ProjectionConst,
    /// A top level const item not part of a trait or impl.
    FreeConst,
    /// An associated const in an inherent `impl`
    InherentConst,
}

impl AliasTermKind {
    pub fn descr(self) -> &'static str {
        match self {
            AliasTermKind::ProjectionTy => "associated type",
            AliasTermKind::ProjectionConst => "associated const",
            AliasTermKind::InherentTy => "inherent associated type",
            AliasTermKind::InherentConst => "inherent associated const",
            AliasTermKind::OpaqueTy => "opaque type",
            AliasTermKind::FreeTy => "type alias",
            AliasTermKind::FreeConst => "unevaluated constant",
            AliasTermKind::UnevaluatedConst => "unevaluated constant",
        }
    }

    pub fn is_type(self) -> bool {
        match self {
            AliasTermKind::ProjectionTy
            | AliasTermKind::InherentTy
            | AliasTermKind::OpaqueTy
            | AliasTermKind::FreeTy => true,

            AliasTermKind::UnevaluatedConst
            | AliasTermKind::ProjectionConst
            | AliasTermKind::InherentConst
            | AliasTermKind::FreeConst => false,
        }
    }
}

impl From<ty::AliasTyKind> for AliasTermKind {
    fn from(value: ty::AliasTyKind) -> Self {
        match value {
            ty::Projection => AliasTermKind::ProjectionTy,
            ty::Opaque => AliasTermKind::OpaqueTy,
            ty::Free => AliasTermKind::FreeTy,
            ty::Inherent => AliasTermKind::InherentTy,
        }
    }
}

/// Represents the unprojected term of a projection goal.
///
/// * For a projection, this would be `<Ty as Trait<...>>::N<...>`.
/// * For an inherent projection, this would be `Ty::N<...>`.
/// * For an opaque type, there is no explicit syntax.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct AliasTerm<I: Interner> {
    /// The parameters of the associated or opaque item.
    ///
    /// For a projection, these are the generic parameters for the trait and the
    /// GAT parameters, if there are any.
    ///
    /// For an inherent projection, they consist of the self type and the GAT parameters,
    /// if there are any.
    ///
    /// For RPIT the generic parameters are for the generics of the function,
    /// while for TAIT it is used for the generic parameters of the alias.
    pub args: I::GenericArgs,

    /// The `DefId` of the `TraitItem` or `ImplItem` for the associated type `N` depending on whether
    /// this is a projection or an inherent projection or the `DefId` of the `OpaqueType` item if
    /// this is an opaque.
    ///
    /// During codegen, `interner.type_of(def_id)` can be used to get the type of the
    /// underlying type if the type is an opaque.
    ///
    /// Note that if this is an associated type, this is not the `DefId` of the
    /// `TraitRef` containing this associated type, which is in `interner.associated_item(def_id).container`,
    /// aka. `interner.parent(def_id)`.
    pub def_id: I::DefId,

    /// This field exists to prevent the creation of `AliasTerm` without using [`AliasTerm::new_from_args`].
    #[derive_where(skip(Debug))]
    _use_alias_term_new_instead: (),
}

impl<I: Interner> AliasTerm<I> {
    pub fn new_from_args(interner: I, def_id: I::DefId, args: I::GenericArgs) -> AliasTerm<I> {
        interner.debug_assert_args_compatible(def_id, args);
        AliasTerm { def_id, args, _use_alias_term_new_instead: () }
    }

    pub fn new(
        interner: I,
        def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> AliasTerm<I> {
        let args = interner.mk_args_from_iter(args.into_iter().map(Into::into));
        Self::new_from_args(interner, def_id, args)
    }

    pub fn expect_ty(self, interner: I) -> ty::AliasTy<I> {
        match self.kind(interner) {
            AliasTermKind::ProjectionTy
            | AliasTermKind::InherentTy
            | AliasTermKind::OpaqueTy
            | AliasTermKind::FreeTy => {}
            AliasTermKind::InherentConst
            | AliasTermKind::FreeConst
            | AliasTermKind::UnevaluatedConst
            | AliasTermKind::ProjectionConst => {
                panic!("Cannot turn `UnevaluatedConst` into `AliasTy`")
            }
        }
        ty::AliasTy { def_id: self.def_id, args: self.args, _use_alias_ty_new_instead: () }
    }

    pub fn kind(self, interner: I) -> AliasTermKind {
        interner.alias_term_kind(self)
    }

    pub fn to_term(self, interner: I) -> I::Term {
        match self.kind(interner) {
            AliasTermKind::ProjectionTy => Ty::new_alias(
                interner,
                ty::AliasTyKind::Projection,
                ty::AliasTy { def_id: self.def_id, args: self.args, _use_alias_ty_new_instead: () },
            )
            .into(),
            AliasTermKind::InherentTy => Ty::new_alias(
                interner,
                ty::AliasTyKind::Inherent,
                ty::AliasTy { def_id: self.def_id, args: self.args, _use_alias_ty_new_instead: () },
            )
            .into(),
            AliasTermKind::OpaqueTy => Ty::new_alias(
                interner,
                ty::AliasTyKind::Opaque,
                ty::AliasTy { def_id: self.def_id, args: self.args, _use_alias_ty_new_instead: () },
            )
            .into(),
            AliasTermKind::FreeTy => Ty::new_alias(
                interner,
                ty::AliasTyKind::Free,
                ty::AliasTy { def_id: self.def_id, args: self.args, _use_alias_ty_new_instead: () },
            )
            .into(),
            AliasTermKind::FreeConst
            | AliasTermKind::InherentConst
            | AliasTermKind::UnevaluatedConst
            | AliasTermKind::ProjectionConst => I::Const::new_unevaluated(
                interner,
                ty::UnevaluatedConst::new(self.def_id, self.args),
            )
            .into(),
        }
    }
}

/// The following methods work only with (trait) associated term projections.
impl<I: Interner> AliasTerm<I> {
    pub fn self_ty(self) -> I::Ty {
        self.args.type_at(0)
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        AliasTerm::new(
            interner,
            self.def_id,
            [self_ty.into()].into_iter().chain(self.args.iter().skip(1)),
        )
    }

    pub fn trait_def_id(self, interner: I) -> I::DefId {
        assert!(
            matches!(
                self.kind(interner),
                AliasTermKind::ProjectionTy | AliasTermKind::ProjectionConst
            ),
            "expected a projection"
        );
        interner.parent(self.def_id)
    }

    /// Extracts the underlying trait reference and own args from this projection.
    /// For example, if this is a projection of `<T as StreamingIterator>::Item<'a>`,
    /// then this function would return a `T: StreamingIterator` trait reference and
    /// `['a]` as the own args.
    pub fn trait_ref_and_own_args(self, interner: I) -> (TraitRef<I>, I::GenericArgsSlice) {
        interner.trait_ref_and_own_args_for_alias(self.def_id, self.args)
    }

    /// Extracts the underlying trait reference from this projection.
    /// For example, if this is a projection of `<T as Iterator>::Item`,
    /// then this function would return a `T: Iterator` trait reference.
    ///
    /// WARNING: This will drop the args for generic associated types
    /// consider calling [Self::trait_ref_and_own_args] to get those
    /// as well.
    pub fn trait_ref(self, interner: I) -> TraitRef<I> {
        self.trait_ref_and_own_args(interner).0
    }

    /// Extract the own args from this projection.
    /// For example, if this is a projection of `<T as StreamingIterator>::Item<'a>`,
    /// then this function would return the slice `['a]` as the own args.
    pub fn own_args(self, interner: I) -> I::GenericArgsSlice {
        self.trait_ref_and_own_args(interner).1
    }
}

/// The following methods work only with inherent associated term projections.
impl<I: Interner> AliasTerm<I> {
    /// Transform the generic parameters to have the given `impl` args as the base and the GAT args on top of that.
    ///
    /// Does the following transformation:
    ///
    /// ```text
    /// [Self, P_0...P_m] -> [I_0...I_n, P_0...P_m]
    ///
    ///     I_i impl args
    ///     P_j GAT args
    /// ```
    pub fn rebase_inherent_args_onto_impl(
        self,
        impl_args: I::GenericArgs,
        interner: I,
    ) -> I::GenericArgs {
        debug_assert!(matches!(
            self.kind(interner),
            AliasTermKind::InherentTy | AliasTermKind::InherentConst
        ));
        interner.mk_args_from_iter(impl_args.iter().chain(self.args.iter().skip(1)))
    }
}

impl<I: Interner> From<ty::AliasTy<I>> for AliasTerm<I> {
    fn from(ty: ty::AliasTy<I>) -> Self {
        AliasTerm { args: ty.args, def_id: ty.def_id, _use_alias_term_new_instead: () }
    }
}

impl<I: Interner> From<ty::UnevaluatedConst<I>> for AliasTerm<I> {
    fn from(ct: ty::UnevaluatedConst<I>) -> Self {
        AliasTerm { args: ct.args, def_id: ct.def, _use_alias_term_new_instead: () }
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
#[derive_where(Clone, Copy, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct ProjectionPredicate<I: Interner> {
    pub projection_term: AliasTerm<I>,
    pub term: I::Term,
}

impl<I: Interner> ProjectionPredicate<I> {
    pub fn self_ty(self) -> I::Ty {
        self.projection_term.self_ty()
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> ProjectionPredicate<I> {
        Self { projection_term: self.projection_term.with_self_ty(interner, self_ty), ..self }
    }

    pub fn trait_def_id(self, interner: I) -> I::DefId {
        self.projection_term.trait_def_id(interner)
    }

    pub fn def_id(self) -> I::DefId {
        self.projection_term.def_id
    }
}

impl<I: Interner> ty::Binder<I, ProjectionPredicate<I>> {
    /// Returns the `DefId` of the trait of the associated item being projected.
    #[inline]
    pub fn trait_def_id(&self, cx: I) -> I::DefId {
        self.skip_binder().projection_term.trait_def_id(cx)
    }

    pub fn term(&self) -> ty::Binder<I, I::Term> {
        self.map_bound(|predicate| predicate.term)
    }

    /// The `DefId` of the `TraitItem` for the associated type.
    ///
    /// Note that this is not the `DefId` of the `TraitRef` containing this
    /// associated type, which is in `tcx.associated_item(projection_def_id()).container`.
    pub fn item_def_id(&self) -> I::DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().projection_term.def_id
    }
}

impl<I: Interner> fmt::Debug for ProjectionPredicate<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProjectionPredicate({:?}, {:?})", self.projection_term, self.term)
    }
}

/// Used by the new solver to normalize an alias. This always expects the `term` to
/// be an unconstrained inference variable which is used as the output.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct NormalizesTo<I: Interner> {
    pub alias: AliasTerm<I>,
    pub term: I::Term,
}

impl<I: Interner> NormalizesTo<I> {
    pub fn self_ty(self) -> I::Ty {
        self.alias.self_ty()
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> NormalizesTo<I> {
        Self { alias: self.alias.with_self_ty(interner, self_ty), ..self }
    }

    pub fn trait_def_id(self, interner: I) -> I::DefId {
        self.alias.trait_def_id(interner)
    }

    pub fn def_id(self) -> I::DefId {
        self.alias.def_id
    }
}

impl<I: Interner> fmt::Debug for NormalizesTo<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NormalizesTo({:?}, {:?})", self.alias, self.term)
    }
}

#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub struct HostEffectPredicate<I: Interner> {
    pub trait_ref: ty::TraitRef<I>,
    pub constness: BoundConstness,
}

impl<I: Interner> HostEffectPredicate<I> {
    pub fn self_ty(self) -> I::Ty {
        self.trait_ref.self_ty()
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        Self { trait_ref: self.trait_ref.with_self_ty(interner, self_ty), ..self }
    }

    pub fn def_id(self) -> I::DefId {
        self.trait_ref.def_id
    }
}

impl<I: Interner> ty::Binder<I, HostEffectPredicate<I>> {
    pub fn def_id(self) -> I::DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().def_id()
    }

    pub fn self_ty(self) -> ty::Binder<I, I::Ty> {
        self.map_bound(|trait_ref| trait_ref.self_ty())
    }

    #[inline]
    pub fn constness(self) -> BoundConstness {
        self.skip_binder().constness
    }
}

/// Encodes that `a` must be a subtype of `b`. The `a_is_expected` flag indicates
/// whether the `a` type is the type that we should label as "expected" when
/// presenting user diagnostics.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct SubtypePredicate<I: Interner> {
    pub a_is_expected: bool,
    pub a: I::Ty,
    pub b: I::Ty,
}

/// Encodes that we have to coerce *from* the `a` type to the `b` type.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct CoercePredicate<I: Interner> {
    pub a: I::Ty,
    pub b: I::Ty,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub enum BoundConstness {
    /// `Type: const Trait`
    ///
    /// A bound is required to be unconditionally const, even in a runtime function.
    Const,
    /// `Type: [const] Trait`
    ///
    /// Requires resolving to const only when we are in a const context.
    Maybe,
}

impl BoundConstness {
    pub fn satisfies(self, goal: BoundConstness) -> bool {
        match (self, goal) {
            (BoundConstness::Const, BoundConstness::Const | BoundConstness::Maybe) => true,
            (BoundConstness::Maybe, BoundConstness::Maybe) => true,
            (BoundConstness::Maybe, BoundConstness::Const) => false,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Const => "const",
            Self::Maybe => "[const]",
        }
    }
}

impl fmt::Display for BoundConstness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const => f.write_str("const"),
            Self::Maybe => f.write_str("[const]"),
        }
    }
}
