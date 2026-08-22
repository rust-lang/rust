use derive_where::derive_where;
use rustc_index::Idx;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash_NoContext};
use rustc_type_ir_macros::{GenericTypeVisitable, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::inherent::*;
use crate::{
    self as ty, Binder, Flags, Interner, Region, TypeFoldable, TypeFolder, TypeSuperFoldable,
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

// FIXME: Just use `BottomUpFolder` for all those self type replacements below
impl<I: Interner> OpaqueHiddenTyBound<I> {
    pub fn iter_self_bounds_for_alias_ty(
        cx: I,
        alias: ty::AliasTy<I>,
    ) -> impl Iterator<Item = Self> {
        struct ReplaceAlias<I: Interner> {
            cx: I,
            alias: ty::AliasTy<I>,
            self_ty: I::Ty,
        }

        impl<I: Interner> TypeFolder<I> for ReplaceAlias<I> {
            fn cx(&self) -> I {
                self.cx
            }

            fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
                if !ty.has_non_rigid_aliases() {
                    return ty;
                }

                if ty::Alias(ty::IsRigid::No, self.alias) == ty.kind() {
                    self.self_ty
                } else {
                    ty.super_fold_with(self)
                }
            }
        }

        let def_id = match alias.kind {
            ty::AliasTyKind::Projection { def_id } => def_id.into(),
            ty::AliasTyKind::Opaque { def_id } => def_id.into(),
            ty::AliasTyKind::Inherent { .. } | ty::AliasTyKind::Free { .. } => unreachable!(),
        };

        cx.item_self_bounds(def_id).iter_instantiated(cx, alias.args).map(move |bound| {
            let bound = bound.skip_normalization();
            let outermost = bound.outer_exclusive_binder().shifted_in(1);
            let bound = Binder::bind_with_vars(
                bound,
                I::BoundVarKinds::from_vars(cx, [ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]),
            );
            OpaqueHiddenTyBound {
                bound: bound.fold_with(&mut ReplaceAlias {
                    cx,
                    alias,
                    self_ty: Ty::new_anon_bound(cx, outermost, ty::BoundVar::new(0)),
                }),
            }
        })
    }

    // FIXME: Comment here with an example for the following case:
    //
    // fn move_forward() -> impl IntoIterator<Item = i32> {
    //     std::iter::empty()
    //         .map(|_: ()| move_forward())
    //         .flatten()
    //         .collect::<Vec<_>>()
    // }
    pub fn opt_unmentioned_projection(
        cx: I,
        existing_bounds: impl IntoIterator<Item = Self>,
        proj: ty::ProjectionPredicate<I>,
    ) -> Option<Self> {
        struct ReplaceTy<I: Interner> {
            cx: I,
            ty: I::Ty,
            self_ty: I::Ty,
        }

        impl<I: Interner> TypeFolder<I> for ReplaceTy<I> {
            fn cx(&self) -> I {
                self.cx
            }

            fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
                if ty == self.ty { self.self_ty } else { ty.super_fold_with(self) }
            }
        }

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
        let outermost = bound.outer_exclusive_binder().shifted_in(1);
        let bound = Binder::bind_with_vars(
            bound,
            I::BoundVarKinds::from_vars(cx, [ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]),
        );
        Some(OpaqueHiddenTyBound {
            bound: bound.fold_with(&mut ReplaceTy {
                cx,
                ty: proj.self_ty(),
                self_ty: Ty::new_anon_bound(cx, outermost, ty::BoundVar::new(0)),
            }),
        })
    }

    pub fn instantiate(self, cx: I, self_ty: I::Ty) -> I::Clause {
        struct ReplaceAnonSelf<I: Interner> {
            cx: I,
            debrujin: ty::DebruijnIndex,
            bound_ty: ty::BoundTy<I>,
            self_ty: I::Ty,
        }

        impl<I: Interner> TypeFolder<I> for ReplaceAnonSelf<I> {
            fn cx(&self) -> I {
                self.cx
            }

            fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
                if !ty.has_vars_bound_at_or_above(self.debrujin) {
                    return ty;
                }

                if ty::Bound(ty::BoundVarIndexKind::Bound(self.debrujin), self.bound_ty)
                    == ty.kind()
                {
                    self.self_ty
                } else {
                    ty.super_fold_with(self)
                }
            }
        }

        let OpaqueHiddenTyBound { bound } = self;

        debug_assert_eq!(
            bound.bound_vars().as_slice(),
            &[ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)]
        );

        let bound = bound.skip_binder();
        debug_assert!(bound.has_escaping_bound_vars());

        let outermost = bound.outer_exclusive_binder();
        let bound = self.bound.skip_binder().fold_with(&mut ReplaceAnonSelf {
            cx,
            debrujin: outermost,
            bound_ty: ty::BoundTy { var: ty::BoundVar::new(0), kind: ty::BoundTyKind::Anon },
            self_ty,
        });
        debug_assert!(!bound.has_escaping_bound_vars());

        bound
    }
}
