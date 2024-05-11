use std::fmt;

use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::inherent::*;
use crate::Interner;

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
