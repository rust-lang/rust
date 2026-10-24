use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};

use crate::{AliasConstKind, AliasTermKind, AliasTyKind, Interner};

/// Represents an alias of a type, constant, or other term-like item.
///
/// * For a projection, this would be `<Ty as Trait<...>>::N<...>`.
/// * For an inherent projection, this would be `Ty::N<...>`.
/// * For an opaque type, there is no explicit syntax.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner, K)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, StableHash_NoContext)
)]
pub struct Alias<I: Interner, K> {
    pub kind: K,

    /// The parameters of the associated, opaque, or constant alias.
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

    /// This field exists to prevent the creation of `Alias` without using the relevant constructor.
    #[derive_where(skip(Debug))]
    #[type_visitable(ignore)]
    #[type_foldable(identity)]
    #[lift(identity)]
    pub(crate) _use_alias_new_instead: (),
}

impl<I: Interner, K: PartialEq> Eq for Alias<I, K> {}

impl<I: Interner, K: Copy> Alias<I, K> {
    pub fn kind(self, _interner: I) -> K {
        self.kind
    }
}

pub type AliasTerm<I> = Alias<I, AliasTermKind<I>>;
pub type AliasTy<I> = Alias<I, AliasTyKind<I>>;
pub type ProjectionAliasTy<I> = Alias<I, <I as Interner>::TraitAssocTyId>;
pub type InherentAliasTy<I> = Alias<I, <I as Interner>::InherentAssocTyId>;
pub type OpaqueAliasTy<I> = Alias<I, <I as Interner>::OpaqueTyId>;
pub type FreeAliasTy<I> = Alias<I, <I as Interner>::FreeTyAliasId>;
pub type AliasConst<I> = Alias<I, AliasConstKind<I>>;
