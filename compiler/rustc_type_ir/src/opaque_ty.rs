use derive_where::derive_where;
use rustc_index::Idx;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash_NoContext};
use rustc_type_ir_macros::{GenericTypeVisitable, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::inherent::*;
use crate::{
    self as ty, Binder, Interner, Region, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt, Upcast,
};

#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct OpaqueTypeKey<I: Interner> {
    pub def_id: I::LocalOpaqueTyId,
    pub args: I::GenericArgs,
}

impl<I: Interner> Eq for OpaqueTypeKey<I> {}

impl<I: Interner> OpaqueTypeKey<I> {
    pub fn iter_captured_args(self, cx: I) -> impl Iterator<Item = (usize, I::GenericArg)> {
        let variances = cx.variances_of(self.def_id.into());
        std::iter::zip(self.args.iter(), variances.iter()).enumerate().filter_map(
            |(i, (arg, v))| match (arg.kind(), v) {
                (_, ty::Invariant) => Some((i, arg)),
                (ty::GenericArgKind::Lifetime(_), ty::Bivariant) => None,
                _ => panic!("unexpected opaque type arg variance"),
            },
        )
    }

    pub fn fold_captured_lifetime_args(
        self,
        cx: I,
        mut f: impl FnMut(Region<I>) -> Region<I>,
    ) -> Self {
        let Self { def_id, args } = self;
        let variances = cx.variances_of(def_id.into());
        let args =
            std::iter::zip(args.iter(), variances.iter()).map(|(arg, v)| match (arg.kind(), v) {
                (ty::GenericArgKind::Lifetime(_), ty::Bivariant) => arg,
                (ty::GenericArgKind::Lifetime(lt), _) => f(lt).into(),
                _ => arg,
            });
        let args = cx.mk_args_from_iter(args);
        Self { def_id, args }
    }
}

/// An item self bound for a hidden type(either an opaque or projection onto another hidden type).
/// This is meant to be instantiated inside the solver into an assumption for a goal with the goal's
/// self ty to support non-defining usages.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, StableHash_NoContext)
)]
pub struct OpaqueHiddenTyBound<I: Interner> {
    bound: Binder<I, I::Clause>,
}

impl<I: Interner> Eq for OpaqueHiddenTyBound<I> {}

impl<I: Interner> OpaqueHiddenTyBound<I> {
    /// Iterate through the item self bounds of a hidden type for either an opaque
    /// or a projection onto another hidden ty.
    pub fn iter_item_self_bounds_for_hidden_ty(
        cx: I,
        alias: ty::AliasTy<I>,
    ) -> impl Iterator<Item = Self> {
        let def_id = match alias.kind {
            ty::AliasTyKind::Projection { def_id } => def_id.into(),
            ty::AliasTyKind::Opaque { def_id } => def_id.into(),
            ty::AliasTyKind::Inherent { .. } | ty::AliasTyKind::Free { .. } => unreachable!(
                "Opaque hidden type should be either an opaque type or projection on another hidden type"
            ),
        };

        let args = alias.args;
        let alias = I::Ty::new_alias(cx, ty::IsRigid::No, alias);
        cx.item_self_bounds(def_id).iter_instantiated(cx, args).map(move |bound| {
            let bound = Binder::bind_with_vars(
                bound
                    .skip_normalization()
                    .fold_with(&mut ReplaceSelfTyWithAnonBound::new(cx, alias)),
                I::BoundVarKinds::from_vars(cx, [ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]),
            );
            OpaqueHiddenTyBound { bound }
        })
    }

    /// If the given `projection` is not mentioned among the given `existing_bounds`,
    /// create one for it.
    ///
    /// This is needed to support the non-defining usages like in the following case:
    ///
    /// ```no_run
    /// fn argument_types() -> impl IntoIterator<Item = i32> {
    ///     argument_types().into_iter().collect::<Vec<_>>()
    /// //                 ^           ^
    /// //                 |           |
    /// //              `{opaque}`     |
    /// //                           `<{opaque} as IntoIterator>::IntoIter`
    /// }
    /// ```
    ///
    /// We need to prove `<{opaque} as IntoIterator>::IntoIter: Iterator` to select the
    /// method `collect()` on it. But as the given bounds in the scope don't mention the
    /// assoc type `IntoIterator::IntoIter` at all, we can't assemble a candidate for
    /// that trait goal. So, we have manually conjure a bound for such unmentioned
    /// projections.
    pub fn opt_unmentioned_projection_bound(
        cx: I,
        existing_bounds: impl IntoIterator<Item = Self>,
        proj: ty::ProjectionClause<I>,
    ) -> Option<Self> {
        let trait_def_id = proj.trait_def_id(cx);
        let mut mentions_trait = false;
        for bound in existing_bounds.into_iter() {
            if bound
                .bound
                .skip_binder()
                .as_projection_clause()
                .is_some_and(|b| b.item_def_id() == proj.def_id())
            {
                // Mentioned already
                return None;
            }

            if bound
                .bound
                .skip_binder()
                .as_trait_clause()
                .is_some_and(|b| b.def_id() == trait_def_id)
            {
                mentions_trait = true;
            }
        }

        if !mentions_trait {
            return None;
        }

        let bound: I::Clause = proj.upcast(cx);
        let bound = Binder::bind_with_vars(
            bound.fold_with(&mut ReplaceSelfTyWithAnonBound::new(cx, proj.self_ty())),
            I::BoundVarKinds::from_vars(cx, [ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]),
        );
        Some(OpaqueHiddenTyBound { bound })
    }

    pub fn instantiate(self, cx: I, self_ty: I::Ty) -> I::Clause {
        let OpaqueHiddenTyBound { bound } = self;

        debug_assert_eq!(
            bound.bound_vars().as_slice(),
            &[ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]
        );
        debug_assert!(bound.skip_binder().has_escaping_bound_vars());

        let bound =
            self.bound.skip_binder().fold_with(&mut ReplaceAnonBoundWithSelfTy::new(cx, self_ty));
        debug_assert!(!bound.has_escaping_bound_vars());

        bound
    }
}

struct ReplaceSelfTyWithAnonBound<I: Interner> {
    cx: I,
    self_ty: I::Ty,
    debruijn: ty::DebruijnIndex,
    bound_var: ty::BoundVar,
}

impl<I: Interner> ReplaceSelfTyWithAnonBound<I> {
    fn new(cx: I, self_ty: I::Ty) -> Self {
        ReplaceSelfTyWithAnonBound {
            cx,
            self_ty,
            debruijn: ty::INNERMOST,
            bound_var: ty::BoundVar::new(0),
        }
    }
}

impl<I: Interner> TypeFolder<I> for ReplaceSelfTyWithAnonBound<I> {
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
        if ty == self.self_ty {
            I::Ty::new_anon_bound(self.cx, self.debruijn, self.bound_var)
        } else {
            ty.super_fold_with(self)
        }
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T>
    where
        T: TypeFoldable<I>,
    {
        self.debruijn.shift_in(1);
        let result = t.super_fold_with(self);
        self.debruijn.shift_out(1);
        result
    }
}

struct ReplaceAnonBoundWithSelfTy<I: Interner> {
    cx: I,
    self_ty: I::Ty,
    debruijn: ty::DebruijnIndex,
    bound_ty: ty::BoundTy<I>,
}

impl<I: Interner> ReplaceAnonBoundWithSelfTy<I> {
    fn new(cx: I, self_ty: I::Ty) -> Self {
        ReplaceAnonBoundWithSelfTy {
            cx,
            self_ty,
            debruijn: ty::INNERMOST,
            bound_ty: ty::BoundTy { var: ty::BoundVar::new(0), kind: ty::BoundTyKind::Anon },
        }
    }
}

impl<I: Interner> TypeFolder<I> for ReplaceAnonBoundWithSelfTy<I> {
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
        let ty = ty.super_fold_with(self);
        if ty::Bound(ty::BoundVarIndexKind::Bound(self.debruijn), self.bound_ty) == ty.kind() {
            self.self_ty
        } else {
            ty
        }
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T>
    where
        T: TypeFoldable<I>,
    {
        self.debruijn.shift_in(1);
        let result = t.super_fold_with(self);
        self.debruijn.shift_out(1);
        result
    }
}
