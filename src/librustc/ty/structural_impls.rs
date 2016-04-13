// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty::subst::{self, VecPerParamSpace};
use traits;
use ty::{self, Lift, TraitRef, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};

use std::rc::Rc;
use syntax::abi;
use syntax::ptr::P;

use hir;

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>> Lift<'tcx> for (A, B) {
    type Lifted = (A::Lifted, B::Lifted);
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| tcx.lift(&self.1).map(|b| (a, b)))
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for [T] {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<Self::Lifted> {
        // type annotation needed to inform `projection_must_outlive`
        let mut result : Vec<<T as Lift<'tcx>>::Lifted>
            = Vec::with_capacity(self.len());
        for x in self {
            if let Some(value) = tcx.lift(x) {
                result.push(value);
            } else {
                return None;
            }
        }
        Some(result)
    }
}

impl<'tcx> Lift<'tcx> for ty::Region {
    type Lifted = Self;
    fn lift_to_tcx(&self, _: &TyCtxt<'tcx>) -> Option<ty::Region> {
        Some(*self)
    }
}

impl<'a, 'tcx> Lift<'tcx> for TraitRef<'a> {
    type Lifted = TraitRef<'tcx>;
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<TraitRef<'tcx>> {
        tcx.lift(&self.substs).map(|substs| TraitRef {
            def_id: self.def_id,
            substs: substs
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitPredicate<'a> {
    type Lifted = ty::TraitPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<ty::TraitPredicate<'tcx>> {
        tcx.lift(&self.trait_ref).map(|trait_ref| ty::TraitPredicate {
            trait_ref: trait_ref
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::EquatePredicate<'a> {
    type Lifted = ty::EquatePredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<ty::EquatePredicate<'tcx>> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| ty::EquatePredicate(a, b))
    }
}

impl<'tcx, A: Copy+Lift<'tcx>, B: Copy+Lift<'tcx>> Lift<'tcx> for ty::OutlivesPredicate<A, B> {
    type Lifted = ty::OutlivesPredicate<A::Lifted, B::Lifted>;
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| ty::OutlivesPredicate(a, b))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionPredicate<'a> {
    type Lifted = ty::ProjectionPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<ty::ProjectionPredicate<'tcx>> {
        tcx.lift(&(self.projection_ty.trait_ref, self.ty)).map(|(trait_ref, ty)| {
            ty::ProjectionPredicate {
                projection_ty: ty::ProjectionTy {
                    trait_ref: trait_ref,
                    item_name: self.projection_ty.item_name
                },
                ty: ty
            }
        })
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::Binder<T> {
    type Lifted = ty::Binder<T::Lifted>;
    fn lift_to_tcx(&self, tcx: &TyCtxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).map(|x| ty::Binder(x))
    }
}

///////////////////////////////////////////////////////////////////////////
// TypeFoldable implementations.
//
// Ideally, each type should invoke `folder.fold_foo(self)` and
// nothing else. In some cases, though, we haven't gotten around to
// adding methods on the `folder` yet, and thus the folding is
// hard-coded here. This is less-flexible, because folders cannot
// override the behavior, but there are a lot of random types and one
// can easily refactor the folding into the TypeFolder trait as
// needed.

macro_rules! CopyImpls {
    ($($ty:ty),+) => {
        $(
            impl<'tcx> TypeFoldable<'tcx> for $ty {
                fn super_fold_with<F:TypeFolder<'tcx>>(&self, _: &mut F) -> $ty {
                    *self
                }

                fn super_visit_with<F: TypeVisitor<'tcx>>(&self, _: &mut F) -> bool {
                    false
                }
            }
        )+
    }
}

CopyImpls! { (), hir::Unsafety, abi::Abi }

impl<'tcx, T:TypeFoldable<'tcx>, U:TypeFoldable<'tcx>> TypeFoldable<'tcx> for (T, U) {
    fn super_fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> (T, U) {
        (self.0.fold_with(folder), self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Option<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.as_ref().map(|t| t.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Rc<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        Rc::new((**self).fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let content: T = (**self).fold_with(folder);
        box content
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Vec<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T:TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::Binder<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::Binder(self.0.fold_with(folder))
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_binder(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor)
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_binder(self)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for P<[T]> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for VecPerParamSpace<T> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.map(|elem| elem.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|elem| elem.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitTy<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::TraitTy {
            principal: self.principal.fold_with(folder),
            bounds: self.bounds.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.principal.visit_with(visitor) || self.bounds.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let sty = match self.sty {
            ty::TyBox(typ) => ty::TyBox(typ.fold_with(folder)),
            ty::TyRawPtr(ref tm) => ty::TyRawPtr(tm.fold_with(folder)),
            ty::TyArray(typ, sz) => ty::TyArray(typ.fold_with(folder), sz),
            ty::TySlice(typ) => ty::TySlice(typ.fold_with(folder)),
            ty::TyEnum(tid, ref substs) => {
                let substs = substs.fold_with(folder);
                ty::TyEnum(tid, folder.tcx().mk_substs(substs))
            }
            ty::TyTrait(ref trait_ty) => ty::TyTrait(trait_ty.fold_with(folder)),
            ty::TyTuple(ref ts) => ty::TyTuple(ts.fold_with(folder)),
            ty::TyFnDef(def_id, substs, ref f) => {
                let substs = substs.fold_with(folder);
                let bfn = f.fold_with(folder);
                ty::TyFnDef(def_id,
                            folder.tcx().mk_substs(substs),
                            folder.tcx().mk_bare_fn(bfn))
            }
            ty::TyFnPtr(ref f) => {
                let bfn = f.fold_with(folder);
                ty::TyFnPtr(folder.tcx().mk_bare_fn(bfn))
            }
            ty::TyRef(r, ref tm) => {
                let r = r.fold_with(folder);
                ty::TyRef(folder.tcx().mk_region(r), tm.fold_with(folder))
            }
            ty::TyStruct(did, ref substs) => {
                let substs = substs.fold_with(folder);
                ty::TyStruct(did, folder.tcx().mk_substs(substs))
            }
            ty::TyClosure(did, ref substs) => {
                ty::TyClosure(did, substs.fold_with(folder))
            }
            ty::TyProjection(ref data) => ty::TyProjection(data.fold_with(folder)),
            ty::TyBool | ty::TyChar | ty::TyStr | ty::TyInt(_) |
            ty::TyUint(_) | ty::TyFloat(_) | ty::TyError | ty::TyInfer(_) |
            ty::TyParam(..) => self.sty.clone(),
        };
        folder.tcx().mk_ty(sty)
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_ty(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match self.sty {
            ty::TyBox(typ) => typ.visit_with(visitor),
            ty::TyRawPtr(ref tm) => tm.visit_with(visitor),
            ty::TyArray(typ, _sz) => typ.visit_with(visitor),
            ty::TySlice(typ) => typ.visit_with(visitor),
            ty::TyEnum(_tid, ref substs) => substs.visit_with(visitor),
            ty::TyTrait(ref trait_ty) => trait_ty.visit_with(visitor),
            ty::TyTuple(ref ts) => ts.visit_with(visitor),
            ty::TyFnDef(_, substs, ref f) => {
                substs.visit_with(visitor) || f.visit_with(visitor)
            }
            ty::TyFnPtr(ref f) => f.visit_with(visitor),
            ty::TyRef(r, ref tm) => r.visit_with(visitor) || tm.visit_with(visitor),
            ty::TyStruct(_did, ref substs) => substs.visit_with(visitor),
            ty::TyClosure(_did, ref substs) => substs.visit_with(visitor),
            ty::TyProjection(ref data) => data.visit_with(visitor),
            ty::TyBool | ty::TyChar | ty::TyStr | ty::TyInt(_) |
            ty::TyUint(_) | ty::TyFloat(_) | ty::TyError | ty::TyInfer(_) |
            ty::TyParam(..) => false,
        }
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_ty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::BareFnTy<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::BareFnTy { sig: self.sig.fold_with(folder),
                       abi: self.abi,
                       unsafety: self.unsafety }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_bare_fn_ty(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.sig.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureTy<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
       ty::ClosureTy {
            sig: self.sig.fold_with(folder),
            unsafety: self.unsafety,
            abi: self.abi,
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_closure_ty(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.sig.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeAndMut<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::TypeAndMut { ty: self.ty.fold_with(folder), mutbl: self.mutbl }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_mt(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnOutput<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::FnConverging(ref ty) => ty::FnConverging(ty.fold_with(folder)),
            ty::FnDiverging => ty::FnDiverging
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_output(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ty::FnConverging(ref ty) => ty.visit_with(visitor),
            ty::FnDiverging => false,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnSig<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::FnSig { inputs: self.inputs.fold_with(folder),
                    output: self.output.fold_with(folder),
                    variadic: self.variadic }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_fn_sig(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.inputs.visit_with(visitor) || self.output.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitRef<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let substs = self.substs.fold_with(folder);
        ty::TraitRef {
            def_id: self.def_id,
            substs: folder.tcx().mk_substs(substs),
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_trait_ref(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ImplHeader<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::ImplHeader {
            impl_def_id: self.impl_def_id,
            self_ty: self.self_ty.fold_with(folder),
            trait_ref: self.trait_ref.map(|t| t.fold_with(folder)),
            predicates: self.predicates.iter().map(|p| p.fold_with(folder)).collect(),
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_impl_header(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.self_ty.visit_with(visitor) ||
            self.trait_ref.map(|r| r.visit_with(visitor)).unwrap_or(false) ||
            self.predicates.iter().any(|p| p.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Region {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> Self {
        *self
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_region(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> bool {
        false
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_region(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for subst::Substs<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        subst::Substs { regions: self.regions.fold_with(folder),
                        types: self.types.fold_with(folder) }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_substs(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.types.visit_with(visitor) || self.regions.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureSubsts<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        let func_substs = self.func_substs.fold_with(folder);
        ty::ClosureSubsts {
            func_substs: folder.tcx().mk_substs(func_substs),
            upvar_tys: self.upvar_tys.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.func_substs.visit_with(visitor) || self.upvar_tys.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ItemSubsts<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::ItemSubsts {
            substs: self.substs.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::AutoRef<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::adjustment::AutoPtr(r, m) => {
                let r = r.fold_with(folder);
                ty::adjustment::AutoPtr(folder.tcx().mk_region(r), m)
            }
            ty::adjustment::AutoUnsafe(m) => ty::adjustment::AutoUnsafe(m)
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_autoref(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ty::adjustment::AutoPtr(r, _m) => r.visit_with(visitor),
            ty::adjustment::AutoUnsafe(_m) => false,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::BuiltinBounds {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> Self {
        *self
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> bool {
        false
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialBounds<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::ExistentialBounds {
            region_bound: self.region_bound.fold_with(folder),
            builtin_bounds: self.builtin_bounds,
            projection_bounds: self.projection_bounds.fold_with(folder),
        }
    }

    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_existential_bounds(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.region_bound.visit_with(visitor) || self.projection_bounds.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeParameterDef<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::TypeParameterDef {
            name: self.name,
            def_id: self.def_id,
            space: self.space,
            index: self.index,
            default: self.default.fold_with(folder),
            default_def_id: self.default_def_id,
            object_lifetime_default: self.object_lifetime_default.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.default.visit_with(visitor) ||
            self.object_lifetime_default.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ObjectLifetimeDefault {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::ObjectLifetimeDefault::Ambiguous =>
                ty::ObjectLifetimeDefault::Ambiguous,

            ty::ObjectLifetimeDefault::BaseDefault =>
                ty::ObjectLifetimeDefault::BaseDefault,

            ty::ObjectLifetimeDefault::Specific(r) =>
                ty::ObjectLifetimeDefault::Specific(r.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ty::ObjectLifetimeDefault::Specific(r) => r.visit_with(visitor),
            _ => false,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::RegionParameterDef {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::RegionParameterDef {
            name: self.name,
            def_id: self.def_id,
            space: self.space,
            index: self.index,
            bounds: self.bounds.fold_with(folder)
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.bounds.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Generics<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::Generics {
            types: self.types.fold_with(folder),
            regions: self.regions.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.types.visit_with(visitor) || self.regions.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::GenericPredicates<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::GenericPredicates {
            predicates: self.predicates.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicates.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::Predicate::Trait(ref a) =>
                ty::Predicate::Trait(a.fold_with(folder)),
            ty::Predicate::Equate(ref binder) =>
                ty::Predicate::Equate(binder.fold_with(folder)),
            ty::Predicate::RegionOutlives(ref binder) =>
                ty::Predicate::RegionOutlives(binder.fold_with(folder)),
            ty::Predicate::TypeOutlives(ref binder) =>
                ty::Predicate::TypeOutlives(binder.fold_with(folder)),
            ty::Predicate::Projection(ref binder) =>
                ty::Predicate::Projection(binder.fold_with(folder)),
            ty::Predicate::WellFormed(data) =>
                ty::Predicate::WellFormed(data.fold_with(folder)),
            ty::Predicate::ClosureKind(closure_def_id, kind) =>
                ty::Predicate::ClosureKind(closure_def_id, kind),
            ty::Predicate::ObjectSafe(trait_def_id) =>
                ty::Predicate::ObjectSafe(trait_def_id),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ty::Predicate::Trait(ref a) => a.visit_with(visitor),
            ty::Predicate::Equate(ref binder) => binder.visit_with(visitor),
            ty::Predicate::RegionOutlives(ref binder) => binder.visit_with(visitor),
            ty::Predicate::TypeOutlives(ref binder) => binder.visit_with(visitor),
            ty::Predicate::Projection(ref binder) => binder.visit_with(visitor),
            ty::Predicate::WellFormed(data) => data.visit_with(visitor),
            ty::Predicate::ClosureKind(_closure_def_id, _kind) => false,
            ty::Predicate::ObjectSafe(_trait_def_id) => false,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionPredicate<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::ProjectionPredicate {
            projection_ty: self.projection_ty.fold_with(folder),
            ty: self.ty.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.projection_ty.visit_with(visitor) || self.ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionTy<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::ProjectionTy {
            trait_ref: self.trait_ref.fold_with(folder),
            item_name: self.item_name,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.trait_ref.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::InstantiatedPredicates<'tcx> {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::InstantiatedPredicates {
            predicates: self.predicates.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicates.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::EquatePredicate<'tcx> {
    fn super_fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::EquatePredicate(self.0.fold_with(folder),
                            self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitPredicate<'tcx> {
    fn super_fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::TraitPredicate {
            trait_ref: self.trait_ref.fold_with(folder)
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.trait_ref.visit_with(visitor)
    }
}

impl<'tcx,T,U> TypeFoldable<'tcx> for ty::OutlivesPredicate<T,U>
    where T : TypeFoldable<'tcx>,
          U : TypeFoldable<'tcx>,
{
    fn super_fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::OutlivesPredicate(self.0.fold_with(folder),
                              self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureUpvar<'tcx> {
    fn super_fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::ClosureUpvar {
            def: self.def,
            span: self.span,
            ty: self.ty.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.ty.visit_with(visitor)
    }
}

impl<'a, 'tcx> TypeFoldable<'tcx> for ty::ParameterEnvironment<'a, 'tcx> where 'tcx: 'a {
    fn super_fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::ParameterEnvironment {
            tcx: self.tcx,
            free_substs: self.free_substs.fold_with(folder),
            implicit_region_bound: self.implicit_region_bound.fold_with(folder),
            caller_bounds: self.caller_bounds.fold_with(folder),
            selection_cache: traits::SelectionCache::new(),
            evaluation_cache: traits::EvaluationCache::new(),
            free_id_outlive: self.free_id_outlive,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.free_substs.visit_with(visitor) ||
            self.implicit_region_bound.visit_with(visitor) ||
            self.caller_bounds.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeScheme<'tcx>  {
    fn super_fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        ty::TypeScheme {
            generics: self.generics.fold_with(folder),
            ty: self.ty.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.generics.visit_with(visitor) || self.ty.visit_with(visitor)
    }
}
