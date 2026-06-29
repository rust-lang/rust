use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};

use crate::inherent::*;
use crate::{self as ty, AliasTerm, Interner};

#[derive_where(Clone, Copy, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, StableHash_NoContext)
)]
pub enum TermKind<I: Interner> {
    Ty(I::Ty),
    Const(I::Const),
}

impl<I: Interner> Eq for TermKind<I> {}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub enum AliasTermKind<I: Interner> {
    /// A projection `<Type as Trait>::AssocType`.
    ///
    /// Can get normalized away if monomorphic enough.
    ///
    /// The `def_id` is the `DefId` of the `TraitItem` for the associated type.
    ///
    /// Note that the `def_id` is not the `DefId` of the `TraitRef` containing this
    /// associated type, which is in `interner.associated_item(def_id).container`,
    /// aka. `interner.parent(def_id)`.
    ProjectionTy { def_id: I::TraitAssocTyId },

    /// An associated type in an inherent `impl`
    ///
    /// The `def_id` is the `DefId` of the `ImplItem` for the associated type.
    InherentTy { def_id: I::InherentAssocTyId },

    /// An opaque type (usually from `impl Trait` in type aliases or function return types)
    ///
    /// `def_id` is the `DefId` of the `OpaqueType` item.
    ///
    /// Can only be normalized away in `PostAnalysis` mode or its defining scope.
    ///
    /// During codegen, `interner.type_of(def_id)` can be used to get the type of the
    /// underlying type if the type is an opaque.
    OpaqueTy { def_id: I::OpaqueTyId },

    /// A type alias that actually checks its trait bounds.
    ///
    /// Currently only used if the type alias references opaque types.
    /// Can always be normalized away.
    FreeTy { def_id: I::FreeTyAliasId },

    /// An anonymous constant.
    AnonConst { def_id: I::AnonConstId },
    /// A const alias coming from an associated const.
    ProjectionConst { def_id: I::TraitAssocConstId },
    /// A top level const item not part of a trait or impl.
    FreeConst { def_id: I::FreeConstAliasId },
    /// An associated const in an inherent `impl`
    InherentConst { def_id: I::InherentAssocConstId },
}

impl<I: Interner> AliasTermKind<I> {
    pub fn descr(self) -> &'static str {
        match self {
            AliasTermKind::ProjectionTy { .. } => "associated type",
            AliasTermKind::ProjectionConst { .. } => "associated const",
            AliasTermKind::InherentTy { .. } => "inherent associated type",
            AliasTermKind::InherentConst { .. } => "inherent associated const",
            AliasTermKind::OpaqueTy { .. } => "opaque type",
            AliasTermKind::FreeTy { .. } => "type alias",
            AliasTermKind::FreeConst { .. } => "const alias",
            AliasTermKind::AnonConst { .. } => "anonymous constant",
        }
    }

    pub fn is_type(self) -> bool {
        match self {
            AliasTermKind::ProjectionTy { .. }
            | AliasTermKind::InherentTy { .. }
            | AliasTermKind::OpaqueTy { .. }
            | AliasTermKind::FreeTy { .. } => true,

            AliasTermKind::AnonConst { .. }
            | AliasTermKind::ProjectionConst { .. }
            | AliasTermKind::InherentConst { .. }
            | AliasTermKind::FreeConst { .. } => false,
        }
    }

    pub fn is_trait_projection(self) -> bool {
        match self {
            AliasTermKind::ProjectionTy { .. } | AliasTermKind::ProjectionConst { .. } => true,
            AliasTermKind::InherentTy { .. }
            | AliasTermKind::OpaqueTy { .. }
            | AliasTermKind::FreeTy { .. }
            | AliasTermKind::AnonConst { .. }
            | AliasTermKind::FreeConst { .. }
            | AliasTermKind::InherentConst { .. } => false,
        }
    }
}

impl<I: Interner> From<ty::AliasTyKind<I>> for AliasTermKind<I> {
    fn from(value: ty::AliasTyKind<I>) -> Self {
        match value {
            ty::Projection { def_id } => AliasTermKind::ProjectionTy { def_id },
            ty::Opaque { def_id } => AliasTermKind::OpaqueTy { def_id },
            ty::Free { def_id } => AliasTermKind::FreeTy { def_id },
            ty::Inherent { def_id } => AliasTermKind::InherentTy { def_id },
        }
    }
}

impl<I: Interner> From<ty::AliasConstKind<I>> for AliasTermKind<I> {
    fn from(value: ty::AliasConstKind<I>) -> Self {
        match value {
            ty::AliasConstKind::Projection { def_id } => AliasTermKind::ProjectionConst { def_id },
            ty::AliasConstKind::Inherent { def_id } => AliasTermKind::InherentConst { def_id },
            ty::AliasConstKind::Free { def_id } => AliasTermKind::FreeConst { def_id },
            ty::AliasConstKind::Anon { def_id } => AliasTermKind::AnonConst { def_id },
        }
    }
}

impl<I: Interner> AliasTerm<I> {
    pub fn new_from_args(
        interner: I,
        kind: AliasTermKind<I>,
        args: I::GenericArgs,
    ) -> AliasTerm<I> {
        if cfg!(debug_assertions) {
            let def_id = match kind {
                AliasTermKind::ProjectionTy { def_id } => def_id.into(),
                AliasTermKind::InherentTy { def_id } => def_id.into(),
                AliasTermKind::OpaqueTy { def_id } => def_id.into(),
                AliasTermKind::FreeTy { def_id } => def_id.into(),
                AliasTermKind::AnonConst { def_id } => def_id.into(),
                AliasTermKind::ProjectionConst { def_id } => def_id.into(),
                AliasTermKind::FreeConst { def_id } => def_id.into(),
                AliasTermKind::InherentConst { def_id } => def_id.into(),
            };
            interner.debug_assert_args_compatible(def_id, args);
        }
        AliasTerm { kind, args, _use_alias_new_instead: () }
    }

    pub fn new(
        interner: I,
        kind: AliasTermKind<I>,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> AliasTerm<I> {
        let args = interner.mk_args_from_iter(args.into_iter().map(Into::into));
        Self::new_from_args(interner, kind, args)
    }

    pub fn new_from_def_id(interner: I, def_id: I::DefId, args: I::GenericArgs) -> AliasTerm<I> {
        let kind = interner.alias_term_kind_from_def_id(def_id);
        Self::new_from_args(interner, kind, args)
    }

    pub fn expect_ty(self) -> ty::AliasTy<I> {
        let kind = match self.kind {
            AliasTermKind::ProjectionTy { def_id } => ty::AliasTyKind::Projection { def_id },
            AliasTermKind::InherentTy { def_id } => ty::AliasTyKind::Inherent { def_id },
            AliasTermKind::OpaqueTy { def_id } => ty::AliasTyKind::Opaque { def_id },
            AliasTermKind::FreeTy { def_id } => ty::AliasTyKind::Free { def_id },
            kind @ (AliasTermKind::InherentConst { .. }
            | AliasTermKind::FreeConst { .. }
            | AliasTermKind::AnonConst { .. }
            | AliasTermKind::ProjectionConst { .. }) => {
                panic!("Cannot turn `{}` into `AliasTy`", kind.descr())
            }
        };
        ty::AliasTy { kind, args: self.args, _use_alias_new_instead: () }
    }

    pub fn expect_ct(self) -> ty::AliasConst<I> {
        let kind = match self.kind {
            AliasTermKind::InherentConst { def_id } => ty::AliasConstKind::Inherent { def_id },
            AliasTermKind::FreeConst { def_id } => ty::AliasConstKind::Free { def_id },
            AliasTermKind::AnonConst { def_id } => ty::AliasConstKind::Anon { def_id },
            AliasTermKind::ProjectionConst { def_id } => ty::AliasConstKind::Projection { def_id },
            kind @ (AliasTermKind::ProjectionTy { .. }
            | AliasTermKind::InherentTy { .. }
            | AliasTermKind::OpaqueTy { .. }
            | AliasTermKind::FreeTy { .. }) => {
                panic!("Cannot turn `{}` into `AliasConst`", kind.descr())
            }
        };
        ty::AliasConst { kind, args: self.args, _use_alias_new_instead: () }
    }

    pub fn to_term(self, interner: I, is_rigid: ty::IsRigid) -> I::Term {
        let alias_ty = |kind| {
            Ty::new_alias(interner, is_rigid, ty::AliasTy::new_from_args(interner, kind, self.args))
                .into()
        };
        let alias_const = |kind| {
            I::Const::new_alias(interner, is_rigid, ty::AliasConst::new(interner, kind, self.args))
                .into()
        };
        match self.kind {
            AliasTermKind::FreeConst { def_id } => alias_const(ty::AliasConstKind::Free { def_id }),
            AliasTermKind::InherentConst { def_id } => {
                alias_const(ty::AliasConstKind::Inherent { def_id })
            }
            AliasTermKind::AnonConst { def_id } => alias_const(ty::AliasConstKind::Anon { def_id }),
            AliasTermKind::ProjectionConst { def_id } => {
                alias_const(ty::AliasConstKind::Projection { def_id })
            }
            AliasTermKind::ProjectionTy { def_id } => alias_ty(ty::Projection { def_id }),
            AliasTermKind::InherentTy { def_id } => alias_ty(ty::Inherent { def_id }),
            AliasTermKind::OpaqueTy { def_id } => alias_ty(ty::Opaque { def_id }),
            AliasTermKind::FreeTy { def_id } => alias_ty(ty::Free { def_id }),
        }
    }

    pub fn with_args(self, interner: I, args: I::GenericArgs) -> Self {
        Self::new_from_args(interner, self.kind, args)
    }

    pub fn expect_projection_ty_def_id(self) -> I::TraitAssocTyId {
        match self.kind {
            AliasTermKind::ProjectionTy { def_id } => def_id,
            kind => panic!("expected projection ty, found {kind:?}"),
        }
    }

    pub fn expect_opaque_ty_def_id(self) -> I::OpaqueTyId {
        match self.kind {
            AliasTermKind::OpaqueTy { def_id } => def_id,
            kind => panic!("expected opaque ty, found {kind:?}"),
        }
    }
}

/// The following methods work only with (trait) associated term projections.
// FIXME: Replace by an impl on Alias<ProjectionAliasTermKind>
impl<I: Interner> AliasTerm<I> {
    pub fn self_ty(self) -> I::Ty {
        self.args.type_at(0)
    }

    pub fn with_replaced_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        AliasTerm::new(
            interner,
            self.kind,
            [self_ty.into()].into_iter().chain(self.args.iter().skip(1)),
        )
    }

    pub fn expect_projection_def_id(self) -> I::TraitAssocTermId {
        match self.kind {
            AliasTermKind::ProjectionTy { def_id } => def_id.into(),
            AliasTermKind::ProjectionConst { def_id } => def_id.into(),
            kind => panic!("expected projection alias, found {kind:?}"),
        }
    }

    pub fn trait_def_id(self, interner: I) -> I::TraitId {
        interner.projection_parent(self.expect_projection_def_id())
    }

    /// Extracts the underlying trait reference and own args from this projection.
    /// For example, if this is a projection of `<T as StreamingIterator>::Item<'a>`,
    /// then this function would return a `T: StreamingIterator` trait reference and
    /// `['a]` as the own args.
    pub fn trait_ref_and_own_args(self, interner: I) -> (ty::TraitRef<I>, I::GenericArgsSlice) {
        interner.trait_ref_and_own_args_for_alias(self.expect_projection_def_id(), self.args)
    }

    /// Extracts the underlying trait reference from this projection.
    /// For example, if this is a projection of `<T as Iterator>::Item`,
    /// then this function would return a `T: Iterator` trait reference.
    ///
    /// WARNING: This will drop the args for generic associated types
    /// consider calling [Self::trait_ref_and_own_args] to get those
    /// as well.
    pub fn trait_ref(self, interner: I) -> ty::TraitRef<I> {
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
// FIXME: Replace by an impl on Alias<InherentAliasTermKind>
impl<I: Interner> AliasTerm<I> {
    pub fn expect_inherent_def_id(self) -> I::InherentAssocTermId {
        match self.kind {
            AliasTermKind::InherentTy { def_id } => def_id.into(),
            AliasTermKind::InherentConst { def_id } => def_id.into(),
            kind => panic!("expected inherent alias, found {kind:?}"),
        }
    }

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
            self.kind,
            AliasTermKind::InherentTy { .. } | AliasTermKind::InherentConst { .. }
        ));
        interner.mk_args_from_iter(impl_args.iter().chain(self.args.iter().skip(1)))
    }
}

/// The following methods work only with free term aliases.
// FIXME: Replace by an impl on Alias<FreeAliasTermKind>
impl<I: Interner> AliasTerm<I> {
    pub fn expect_free_def_id(self) -> I::FreeTermAliasId {
        match self.kind {
            AliasTermKind::FreeTy { def_id } => def_id.into(),
            AliasTermKind::FreeConst { def_id } => def_id.into(),
            kind => panic!("expected free alias, found {kind:?}"),
        }
    }
}

impl<I: Interner> From<ty::AliasTy<I>> for AliasTerm<I> {
    fn from(ty: ty::AliasTy<I>) -> Self {
        AliasTerm { args: ty.args, kind: AliasTermKind::from(ty.kind), _use_alias_new_instead: () }
    }
}

impl<I: Interner> From<ty::AliasConst<I>> for AliasTerm<I> {
    fn from(ty: ty::AliasConst<I>) -> Self {
        AliasTerm { args: ty.args, kind: AliasTermKind::from(ty.kind), _use_alias_new_instead: () }
    }
}
