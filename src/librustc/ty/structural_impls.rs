// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::type_variable;
use ty::{self, Lift, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};

use std::rc::Rc;
use syntax::abi;
use syntax::ptr::P;

use hir;

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>> Lift<'tcx> for (A, B) {
    type Lifted = (A::Lifted, B::Lifted);
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| tcx.lift(&self.1).map(|b| (a, b)))
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Option<T> {
    type Lifted = Option<T::Lifted>;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            Some(ref x) => tcx.lift(x).map(Some),
            None => Some(None)
        }
    }
}

impl<'tcx, T: Lift<'tcx>, E: Lift<'tcx>> Lift<'tcx> for Result<T, E> {
    type Lifted = Result<T::Lifted, E::Lifted>;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            Ok(ref x) => tcx.lift(x).map(Ok),
            Err(ref e) => tcx.lift(e).map(Err)
        }
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for [T] {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
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

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for Vec<T> {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self[..])
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitRef<'a> {
    type Lifted = ty::TraitRef<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| ty::TraitRef {
            def_id: self.def_id,
            substs: substs
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialTraitRef<'a> {
    type Lifted = ty::ExistentialTraitRef<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| ty::ExistentialTraitRef {
            def_id: self.def_id,
            substs: substs
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitPredicate<'a> {
    type Lifted = ty::TraitPredicate<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<ty::TraitPredicate<'tcx>> {
        tcx.lift(&self.trait_ref).map(|trait_ref| ty::TraitPredicate {
            trait_ref: trait_ref
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::EquatePredicate<'a> {
    type Lifted = ty::EquatePredicate<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<ty::EquatePredicate<'tcx>> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| ty::EquatePredicate(a, b))
    }
}

impl<'tcx, A: Copy+Lift<'tcx>, B: Copy+Lift<'tcx>> Lift<'tcx> for ty::OutlivesPredicate<A, B> {
    type Lifted = ty::OutlivesPredicate<A::Lifted, B::Lifted>;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| ty::OutlivesPredicate(a, b))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionTy<'a> {
    type Lifted = ty::ProjectionTy<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<ty::ProjectionTy<'tcx>> {
        tcx.lift(&self.trait_ref).map(|trait_ref| {
            ty::ProjectionTy {
                trait_ref: trait_ref,
                item_name: self.item_name
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionPredicate<'a> {
    type Lifted = ty::ProjectionPredicate<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<ty::ProjectionPredicate<'tcx>> {
        tcx.lift(&(self.projection_ty, self.ty)).map(|(projection_ty, ty)| {
            ty::ProjectionPredicate {
                projection_ty: projection_ty,
                ty: ty
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialProjection<'a> {
    type Lifted = ty::ExistentialProjection<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.trait_ref, self.ty)).map(|(trait_ref, ty)| {
            ty::ExistentialProjection {
                trait_ref: trait_ref,
                item_name: self.item_name,
                ty: ty
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::Predicate<'a> {
    type Lifted = ty::Predicate<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::Predicate::Trait(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::Trait)
            }
            ty::Predicate::Equate(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::Equate)
            }
            ty::Predicate::RegionOutlives(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::RegionOutlives)
            }
            ty::Predicate::TypeOutlives(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::TypeOutlives)
            }
            ty::Predicate::Projection(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::Projection)
            }
            ty::Predicate::WellFormed(ty) => {
                tcx.lift(&ty).map(ty::Predicate::WellFormed)
            }
            ty::Predicate::ClosureKind(closure_def_id, kind) => {
                Some(ty::Predicate::ClosureKind(closure_def_id, kind))
            }
            ty::Predicate::ObjectSafe(trait_def_id) => {
                Some(ty::Predicate::ObjectSafe(trait_def_id))
            }
        }
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::Binder<T> {
    type Lifted = ty::Binder<T::Lifted>;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).map(|x| ty::Binder(x))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ClosureSubsts<'a> {
    type Lifted = ty::ClosureSubsts<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.func_substs, self.upvar_tys)).map(|(substs, upvar_tys)| {
            ty::ClosureSubsts {
                func_substs: substs,
                upvar_tys: upvar_tys
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ItemSubsts<'a> {
    type Lifted = ty::ItemSubsts<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| {
            ty::ItemSubsts {
                substs: substs
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::AutoRef<'a> {
    type Lifted = ty::adjustment::AutoRef<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::adjustment::AutoPtr(r, m) => {
                tcx.lift(&r).map(|r| ty::adjustment::AutoPtr(r, m))
            }
            ty::adjustment::AutoUnsafe(m) => {
                Some(ty::adjustment::AutoUnsafe(m))
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::FnSig<'a> {
    type Lifted = ty::FnSig<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.inputs[..]).and_then(|inputs| {
            tcx.lift(&self.output).map(|output| {
                ty::FnSig {
                    inputs: inputs,
                    output: output,
                    variadic: self.variadic
                }
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ClosureTy<'a> {
    type Lifted = ty::ClosureTy<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.sig).map(|sig| {
            ty::ClosureTy {
                sig: sig,
                unsafety: self.unsafety,
                abi: self.abi
            }
        })
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for ty::error::ExpectedFound<T> {
    type Lifted = ty::error::ExpectedFound<T::Lifted>;
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.expected).and_then(|expected| {
            tcx.lift(&self.found).map(|found| {
                ty::error::ExpectedFound {
                    expected: expected,
                    found: found
                }
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for type_variable::Default<'a> {
    type Lifted = type_variable::Default<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.ty).map(|ty| {
            type_variable::Default {
                ty: ty,
                origin_span: self.origin_span,
                def_id: self.def_id
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::error::TypeError<'a> {
    type Lifted = ty::error::TypeError<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        use ty::error::TypeError::*;

        Some(match *self {
            Mismatch => Mismatch,
            UnsafetyMismatch(x) => UnsafetyMismatch(x),
            AbiMismatch(x) => AbiMismatch(x),
            Mutability => Mutability,
            BoxMutability => BoxMutability,
            PtrMutability => PtrMutability,
            RefMutability => RefMutability,
            VecMutability => VecMutability,
            TupleSize(x) => TupleSize(x),
            FixedArraySize(x) => FixedArraySize(x),
            TyParamSize(x) => TyParamSize(x),
            ArgCount => ArgCount,
            RegionsDoesNotOutlive(a, b) => {
                return tcx.lift(&(a, b)).map(|(a, b)| RegionsDoesNotOutlive(a, b))
            }
            RegionsNotSame(a, b) => {
                return tcx.lift(&(a, b)).map(|(a, b)| RegionsNotSame(a, b))
            }
            RegionsNoOverlap(a, b) => {
                return tcx.lift(&(a, b)).map(|(a, b)| RegionsNoOverlap(a, b))
            }
            RegionsInsufficientlyPolymorphic(a, b) => {
                return tcx.lift(&b).map(|b| RegionsInsufficientlyPolymorphic(a, b))
            }
            RegionsOverlyPolymorphic(a, b) => {
                return tcx.lift(&b).map(|b| RegionsOverlyPolymorphic(a, b))
            }
            IntegerAsChar => IntegerAsChar,
            IntMismatch(x) => IntMismatch(x),
            FloatMismatch(x) => FloatMismatch(x),
            Traits(x) => Traits(x),
            BuiltinBoundsMismatch(x) => BuiltinBoundsMismatch(x),
            VariadicMismatch(x) => VariadicMismatch(x),
            CyclicTy => CyclicTy,
            ConvergenceMismatch(x) => ConvergenceMismatch(x),
            ProjectionNameMismatched(x) => ProjectionNameMismatched(x),
            ProjectionBoundsLength(x) => ProjectionBoundsLength(x),

            Sorts(ref x) => return tcx.lift(x).map(Sorts),
            TyParamDefaultMismatch(ref x) => {
                return tcx.lift(x).map(TyParamDefaultMismatch)
            }
        })
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
                fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, _: &mut F) -> $ty {
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
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> (T, U) {
        (self.0.fold_with(folder), self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Option<T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        self.as_ref().map(|t| t.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Rc<T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        Rc::new((**self).fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let content: T = (**self).fold_with(folder);
        box content
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        (**self).visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Vec<T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx, T:TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::Binder<T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::Binder(self.0.fold_with(folder))
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitObject<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::TraitObject {
            principal: self.principal.fold_with(folder),
            region_bound: self.region_bound.fold_with(folder),
            builtin_bounds: self.builtin_bounds,
            projection_bounds: self.projection_bounds.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.principal.visit_with(visitor) ||
        self.region_bound.visit_with(visitor) ||
        self.projection_bounds.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx [Ty<'tcx>] {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let tys = self.iter().map(|t| t.fold_with(folder)).collect();
        folder.tcx().mk_type_list(tys)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let sty = match self.sty {
            ty::TyBox(typ) => ty::TyBox(typ.fold_with(folder)),
            ty::TyRawPtr(tm) => ty::TyRawPtr(tm.fold_with(folder)),
            ty::TyArray(typ, sz) => ty::TyArray(typ.fold_with(folder), sz),
            ty::TySlice(typ) => ty::TySlice(typ.fold_with(folder)),
            ty::TyEnum(tid, substs) => ty::TyEnum(tid, substs.fold_with(folder)),
            ty::TyTrait(ref trait_ty) => ty::TyTrait(trait_ty.fold_with(folder)),
            ty::TyTuple(ts) => ty::TyTuple(ts.fold_with(folder)),
            ty::TyFnDef(def_id, substs, f) => {
                ty::TyFnDef(def_id,
                            substs.fold_with(folder),
                            f.fold_with(folder))
            }
            ty::TyFnPtr(f) => ty::TyFnPtr(f.fold_with(folder)),
            ty::TyRef(ref r, tm) => {
                ty::TyRef(r.fold_with(folder), tm.fold_with(folder))
            }
            ty::TyStruct(did, substs) => ty::TyStruct(did, substs.fold_with(folder)),
            ty::TyUnion(did, substs) => ty::TyUnion(did, substs.fold_with(folder)),
            ty::TyClosure(did, substs) => ty::TyClosure(did, substs.fold_with(folder)),
            ty::TyProjection(ref data) => ty::TyProjection(data.fold_with(folder)),
            ty::TyAnon(did, substs) => ty::TyAnon(did, substs.fold_with(folder)),
            ty::TyBool | ty::TyChar | ty::TyStr | ty::TyInt(_) |
            ty::TyUint(_) | ty::TyFloat(_) | ty::TyError | ty::TyInfer(_) |
            ty::TyParam(..) | ty::TyNever => self.sty.clone(),
        };
        folder.tcx().mk_ty(sty)
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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
            ty::TyTuple(ts) => ts.visit_with(visitor),
            ty::TyFnDef(_, substs, ref f) => {
                substs.visit_with(visitor) || f.visit_with(visitor)
            }
            ty::TyFnPtr(ref f) => f.visit_with(visitor),
            ty::TyRef(r, ref tm) => r.visit_with(visitor) || tm.visit_with(visitor),
            ty::TyStruct(_did, ref substs) => substs.visit_with(visitor),
            ty::TyUnion(_did, ref substs) => substs.visit_with(visitor),
            ty::TyClosure(_did, ref substs) => substs.visit_with(visitor),
            ty::TyProjection(ref data) => data.visit_with(visitor),
            ty::TyAnon(_, ref substs) => substs.visit_with(visitor),
            ty::TyBool | ty::TyChar | ty::TyStr | ty::TyInt(_) |
            ty::TyUint(_) | ty::TyFloat(_) | ty::TyError | ty::TyInfer(_) |
            ty::TyParam(..) | ty::TyNever => false,
        }
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_ty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::BareFnTy<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let fty = ty::BareFnTy {
            sig: self.sig.fold_with(folder),
            abi: self.abi,
            unsafety: self.unsafety
        };
        folder.tcx().mk_bare_fn(fty)
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_bare_fn_ty(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.sig.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureTy<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
       ty::ClosureTy {
            sig: self.sig.fold_with(folder),
            unsafety: self.unsafety,
            abi: self.abi,
        }
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_closure_ty(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.sig.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeAndMut<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::TypeAndMut { ty: self.ty.fold_with(folder), mutbl: self.mutbl }
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_mt(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnSig<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::FnSig { inputs: self.inputs.fold_with(folder),
                    output: self.output.fold_with(folder),
                    variadic: self.variadic }
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_fn_sig(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.inputs.visit_with(visitor) || self.output.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitRef<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::TraitRef {
            def_id: self.def_id,
            substs: self.substs.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialTraitRef<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ExistentialTraitRef {
            def_id: self.def_id,
            substs: self.substs.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ImplHeader<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ImplHeader {
            impl_def_id: self.impl_def_id,
            self_ty: self.self_ty.fold_with(folder),
            trait_ref: self.trait_ref.map(|t| t.fold_with(folder)),
            predicates: self.predicates.iter().map(|p| p.fold_with(folder)).collect(),
        }
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_impl_header(self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.self_ty.visit_with(visitor) ||
            self.trait_ref.map(|r| r.visit_with(visitor)).unwrap_or(false) ||
            self.predicates.iter().any(|p| p.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Region {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, _folder: &mut F) -> Self {
        *self
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_region(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> bool {
        false
    }

    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        visitor.visit_region(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureSubsts<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ClosureSubsts {
            func_substs: self.func_substs.fold_with(folder),
            upvar_tys: self.upvar_tys.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.func_substs.visit_with(visitor) || self.upvar_tys.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ItemSubsts<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ItemSubsts {
            substs: self.substs.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::AutoRef<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::adjustment::AutoPtr(ref r, m) => {
                ty::adjustment::AutoPtr(r.fold_with(folder), m)
            }
            ty::adjustment::AutoUnsafe(m) => ty::adjustment::AutoUnsafe(m)
        }
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, _folder: &mut F) -> Self {
        *self
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _visitor: &mut V) -> bool {
        false
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeParameterDef<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::TypeParameterDef {
            name: self.name,
            def_id: self.def_id,
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

impl<'tcx> TypeFoldable<'tcx> for ty::ObjectLifetimeDefault<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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

impl<'tcx> TypeFoldable<'tcx> for ty::RegionParameterDef<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::RegionParameterDef {
            name: self.name,
            def_id: self.def_id,
            index: self.index,
            bounds: self.bounds.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.bounds.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ProjectionPredicate {
            projection_ty: self.projection_ty.fold_with(folder),
            ty: self.ty.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.projection_ty.visit_with(visitor) || self.ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialProjection<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ExistentialProjection {
            trait_ref: self.trait_ref.fold_with(folder),
            item_name: self.item_name,
            ty: self.ty.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.trait_ref.visit_with(visitor) || self.ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionTy<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::InstantiatedPredicates {
            predicates: self.predicates.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicates.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::EquatePredicate<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::EquatePredicate(self.0.fold_with(folder),
                            self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitPredicate<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::OutlivesPredicate(self.0.fold_with(folder),
                              self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureUpvar<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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

impl<'tcx> TypeFoldable<'tcx> for ty::ParameterEnvironment<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ParameterEnvironment {
            free_substs: self.free_substs.fold_with(folder),
            implicit_region_bound: self.implicit_region_bound.fold_with(folder),
            caller_bounds: self.caller_bounds.fold_with(folder),
            free_id_outlive: self.free_id_outlive,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.free_substs.visit_with(visitor) ||
            self.implicit_region_bound.visit_with(visitor) ||
            self.caller_bounds.visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::error::ExpectedFound<T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::error::ExpectedFound {
            expected: self.expected.fold_with(folder),
            found: self.found.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.expected.visit_with(visitor) || self.found.visit_with(visitor)
    }
}
