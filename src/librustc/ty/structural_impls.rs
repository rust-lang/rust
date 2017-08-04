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
use middle::const_val::{ConstVal, ConstAggregate};
use ty::{self, Lift, Ty, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use rustc_data_structures::accumulate_vec::AccumulateVec;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};

use std::rc::Rc;
use syntax::abi;

use hir;

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>> Lift<'tcx> for (A, B) {
    type Lifted = (A::Lifted, B::Lifted);
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| tcx.lift(&self.1).map(|b| (a, b)))
    }
}

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>, C: Lift<'tcx>> Lift<'tcx> for (A, B, C) {
    type Lifted = (A::Lifted, B::Lifted, C::Lifted);
    fn lift_to_tcx<'a, 'gcx>(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| {
            tcx.lift(&self.1).and_then(|b| tcx.lift(&self.2).map(|c| (a, b, c)))
        })
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
            substs,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialTraitRef<'a> {
    type Lifted = ty::ExistentialTraitRef<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| ty::ExistentialTraitRef {
            def_id: self.def_id,
            substs,
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitPredicate<'a> {
    type Lifted = ty::TraitPredicate<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<ty::TraitPredicate<'tcx>> {
        tcx.lift(&self.trait_ref).map(|trait_ref| ty::TraitPredicate {
            trait_ref,
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

impl<'a, 'tcx> Lift<'tcx> for ty::SubtypePredicate<'a> {
    type Lifted = ty::SubtypePredicate<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>)
                             -> Option<ty::SubtypePredicate<'tcx>> {
        tcx.lift(&(self.a, self.b)).map(|(a, b)| ty::SubtypePredicate {
            a_is_expected: self.a_is_expected,
            a,
            b,
        })
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
        tcx.lift(&self.substs).map(|substs| {
            ty::ProjectionTy {
                item_def_id: self.item_def_id,
                substs,
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
                projection_ty,
                ty,
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ExistentialProjection<'a> {
    type Lifted = ty::ExistentialProjection<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.substs).map(|substs| {
            ty::ExistentialProjection {
                substs,
                ty: tcx.lift(&self.ty).expect("type must lift when substs do"),
                item_def_id: self.item_def_id,
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
            ty::Predicate::Subtype(ref binder) => {
                tcx.lift(binder).map(ty::Predicate::Subtype)
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
        tcx.lift(&self.substs).map(|substs| {
            ty::ClosureSubsts { substs: substs }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::GeneratorInterior<'a> {
    type Lifted = ty::GeneratorInterior<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.witness).map(|witness| {
            ty::GeneratorInterior { witness }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::Adjustment<'a> {
    type Lifted = ty::adjustment::Adjustment<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.kind).and_then(|kind| {
            tcx.lift(&self.target).map(|target| {
                ty::adjustment::Adjustment { kind, target }
            })
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::Adjust<'a> {
    type Lifted = ty::adjustment::Adjust<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::adjustment::Adjust::NeverToAny =>
                Some(ty::adjustment::Adjust::NeverToAny),
            ty::adjustment::Adjust::ReifyFnPointer =>
                Some(ty::adjustment::Adjust::ReifyFnPointer),
            ty::adjustment::Adjust::UnsafeFnPointer =>
                Some(ty::adjustment::Adjust::UnsafeFnPointer),
            ty::adjustment::Adjust::ClosureFnPointer =>
                Some(ty::adjustment::Adjust::ClosureFnPointer),
            ty::adjustment::Adjust::MutToConstPointer =>
                Some(ty::adjustment::Adjust::MutToConstPointer),
            ty::adjustment::Adjust::Unsize =>
                Some(ty::adjustment::Adjust::Unsize),
            ty::adjustment::Adjust::Deref(ref overloaded) => {
                tcx.lift(overloaded).map(ty::adjustment::Adjust::Deref)
            }
            ty::adjustment::Adjust::Borrow(ref autoref) => {
                tcx.lift(autoref).map(ty::adjustment::Adjust::Borrow)
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::OverloadedDeref<'a> {
    type Lifted = ty::adjustment::OverloadedDeref<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.region).map(|region| {
            ty::adjustment::OverloadedDeref {
                region,
                mutbl: self.mutbl,
            }
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::adjustment::AutoBorrow<'a> {
    type Lifted = ty::adjustment::AutoBorrow<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        match *self {
            ty::adjustment::AutoBorrow::Ref(r, m) => {
                tcx.lift(&r).map(|r| ty::adjustment::AutoBorrow::Ref(r, m))
            }
            ty::adjustment::AutoBorrow::RawPtr(m) => {
                Some(ty::adjustment::AutoBorrow::RawPtr(m))
            }
        }
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::GenSig<'a> {
    type Lifted = ty::GenSig<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.yield_ty, self.return_ty))
            .map(|(yield_ty, return_ty)| {
                ty::GenSig {
                    yield_ty,
                    return_ty,
                }
            })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::FnSig<'a> {
    type Lifted = ty::FnSig<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.inputs_and_output).map(|x| {
            ty::FnSig {
                inputs_and_output: x,
                variadic: self.variadic,
                unsafety: self.unsafety,
                abi: self.abi,
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
                    expected,
                    found,
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
                ty,
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
            TupleSize(x) => TupleSize(x),
            FixedArraySize(x) => FixedArraySize(x),
            ArgCount => ArgCount,
            RegionsDoesNotOutlive(a, b) => {
                return tcx.lift(&(a, b)).map(|(a, b)| RegionsDoesNotOutlive(a, b))
            }
            RegionsInsufficientlyPolymorphic(a, b) => {
                return tcx.lift(&b).map(|b| RegionsInsufficientlyPolymorphic(a, b))
            }
            RegionsOverlyPolymorphic(a, b) => {
                return tcx.lift(&b).map(|b| RegionsOverlyPolymorphic(a, b))
            }
            IntMismatch(x) => IntMismatch(x),
            FloatMismatch(x) => FloatMismatch(x),
            Traits(x) => Traits(x),
            VariadicMismatch(x) => VariadicMismatch(x),
            CyclicTy => CyclicTy,
            ProjectionMismatched(x) => ProjectionMismatched(x),
            ProjectionBoundsLength(x) => ProjectionBoundsLength(x),

            Sorts(ref x) => return tcx.lift(x).map(Sorts),
            TyParamDefaultMismatch(ref x) => {
                return tcx.lift(x).map(TyParamDefaultMismatch)
            }
            ExistentialMismatch(ref x) => return tcx.lift(x).map(ExistentialMismatch)
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

CopyImpls! { (), hir::Unsafety, abi::Abi, hir::def_id::DefId, ::mir::Local }

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

impl<'tcx> TypeFoldable<'tcx> for ty::ParamEnv<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ParamEnv {
            reveal: self.reveal,
            caller_bounds: self.caller_bounds.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        let &ty::ParamEnv { reveal: _, ref caller_bounds } = self;
        caller_bounds.super_visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Slice<ty::ExistentialPredicate<'tcx>> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter().map(|p| p.fold_with(folder)).collect::<AccumulateVec<[_; 8]>>();
        folder.tcx().intern_existential_predicates(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|p| p.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialPredicate<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self  {
        use ty::ExistentialPredicate::*;
        match *self {
            Trait(ref tr) => Trait(tr.fold_with(folder)),
            Projection(ref p) => Projection(p.fold_with(folder)),
            AutoTrait(did) => AutoTrait(did),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ty::ExistentialPredicate::Trait(ref tr) => tr.visit_with(visitor),
            ty::ExistentialPredicate::Projection(ref p) => p.visit_with(visitor),
            ty::ExistentialPredicate::AutoTrait(_) => false,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Slice<Ty<'tcx>> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter().map(|t| t.fold_with(folder)).collect::<AccumulateVec<[_; 8]>>();
        folder.tcx().intern_type_list(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let sty = match self.sty {
            ty::TyRawPtr(tm) => ty::TyRawPtr(tm.fold_with(folder)),
            ty::TyArray(typ, sz) => ty::TyArray(typ.fold_with(folder), sz),
            ty::TySlice(typ) => ty::TySlice(typ.fold_with(folder)),
            ty::TyAdt(tid, substs) => ty::TyAdt(tid, substs.fold_with(folder)),
            ty::TyDynamic(ref trait_ty, ref region) =>
                ty::TyDynamic(trait_ty.fold_with(folder), region.fold_with(folder)),
            ty::TyTuple(ts, defaulted) => ty::TyTuple(ts.fold_with(folder), defaulted),
            ty::TyFnDef(def_id, substs) => {
                ty::TyFnDef(def_id, substs.fold_with(folder))
            }
            ty::TyFnPtr(f) => ty::TyFnPtr(f.fold_with(folder)),
            ty::TyRef(ref r, tm) => {
                ty::TyRef(r.fold_with(folder), tm.fold_with(folder))
            }
            ty::TyGenerator(did, substs, interior) => {
                ty::TyGenerator(did, substs.fold_with(folder), interior.fold_with(folder))
            }
            ty::TyClosure(did, substs) => ty::TyClosure(did, substs.fold_with(folder)),
            ty::TyProjection(ref data) => ty::TyProjection(data.fold_with(folder)),
            ty::TyAnon(did, substs) => ty::TyAnon(did, substs.fold_with(folder)),
            ty::TyBool | ty::TyChar | ty::TyStr | ty::TyInt(_) |
            ty::TyUint(_) | ty::TyFloat(_) | ty::TyError | ty::TyInfer(_) |
            ty::TyParam(..) | ty::TyNever => return self
        };

        if self.sty == sty {
            self
        } else {
            folder.tcx().mk_ty(sty)
        }
    }

    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        folder.fold_ty(*self)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match self.sty {
            ty::TyRawPtr(ref tm) => tm.visit_with(visitor),
            ty::TyArray(typ, _sz) => typ.visit_with(visitor),
            ty::TySlice(typ) => typ.visit_with(visitor),
            ty::TyAdt(_, substs) => substs.visit_with(visitor),
            ty::TyDynamic(ref trait_ty, ref reg) =>
                trait_ty.visit_with(visitor) || reg.visit_with(visitor),
            ty::TyTuple(ts, _) => ts.visit_with(visitor),
            ty::TyFnDef(_, substs) => substs.visit_with(visitor),
            ty::TyFnPtr(ref f) => f.visit_with(visitor),
            ty::TyRef(r, ref tm) => r.visit_with(visitor) || tm.visit_with(visitor),
            ty::TyGenerator(_did, ref substs, ref interior) => {
                substs.visit_with(visitor) || interior.visit_with(visitor)
            }
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

impl<'tcx> TypeFoldable<'tcx> for ty::TypeAndMut<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::TypeAndMut { ty: self.ty.fold_with(folder), mutbl: self.mutbl }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::GenSig<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::GenSig {
            yield_ty: self.yield_ty.fold_with(folder),
            return_ty: self.return_ty.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.yield_ty.visit_with(visitor) ||
        self.return_ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnSig<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let inputs_and_output = self.inputs_and_output.fold_with(folder);
        ty::FnSig {
            inputs_and_output: folder.tcx().intern_type_list(&inputs_and_output),
            variadic: self.variadic,
            unsafety: self.unsafety,
            abi: self.abi,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.inputs().iter().any(|i| i.visit_with(visitor)) ||
        self.output().visit_with(visitor)
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

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.self_ty.visit_with(visitor) ||
            self.trait_ref.map(|r| r.visit_with(visitor)).unwrap_or(false) ||
            self.predicates.iter().any(|p| p.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Region<'tcx> {
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
            substs: self.substs.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::GeneratorInterior<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::GeneratorInterior::new(self.witness.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.witness.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::Adjustment<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::adjustment::Adjustment {
            kind: self.kind.fold_with(folder),
            target: self.target.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.kind.visit_with(visitor) ||
        self.target.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::Adjust<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::adjustment::Adjust::NeverToAny |
            ty::adjustment::Adjust::ReifyFnPointer |
            ty::adjustment::Adjust::UnsafeFnPointer |
            ty::adjustment::Adjust::ClosureFnPointer |
            ty::adjustment::Adjust::MutToConstPointer |
            ty::adjustment::Adjust::Unsize => self.clone(),
            ty::adjustment::Adjust::Deref(ref overloaded) => {
                ty::adjustment::Adjust::Deref(overloaded.fold_with(folder))
            }
            ty::adjustment::Adjust::Borrow(ref autoref) => {
                ty::adjustment::Adjust::Borrow(autoref.fold_with(folder))
            }
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ty::adjustment::Adjust::NeverToAny |
            ty::adjustment::Adjust::ReifyFnPointer |
            ty::adjustment::Adjust::UnsafeFnPointer |
            ty::adjustment::Adjust::ClosureFnPointer |
            ty::adjustment::Adjust::MutToConstPointer |
            ty::adjustment::Adjust::Unsize => false,
            ty::adjustment::Adjust::Deref(ref overloaded) => {
                overloaded.visit_with(visitor)
            }
            ty::adjustment::Adjust::Borrow(ref autoref) => {
                autoref.visit_with(visitor)
            }
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::OverloadedDeref<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::adjustment::OverloadedDeref {
            region: self.region.fold_with(folder),
            mutbl: self.mutbl,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.region.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::AutoBorrow<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::adjustment::AutoBorrow::Ref(ref r, m) => {
                ty::adjustment::AutoBorrow::Ref(r.fold_with(folder), m)
            }
            ty::adjustment::AutoBorrow::RawPtr(m) => ty::adjustment::AutoBorrow::RawPtr(m)
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ty::adjustment::AutoBorrow::Ref(r, _m) => r.visit_with(visitor),
            ty::adjustment::AutoBorrow::RawPtr(_m) => false,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::GenericPredicates<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::GenericPredicates {
            parent: self.parent,
            predicates: self.predicates.fold_with(folder),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.predicates.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Slice<ty::Predicate<'tcx>> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let v = self.iter().map(|p| p.fold_with(folder)).collect::<AccumulateVec<[_; 8]>>();
        folder.tcx().intern_predicates(&v)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|p| p.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ty::Predicate::Trait(ref a) =>
                ty::Predicate::Trait(a.fold_with(folder)),
            ty::Predicate::Equate(ref binder) =>
                ty::Predicate::Equate(binder.fold_with(folder)),
            ty::Predicate::Subtype(ref binder) =>
                ty::Predicate::Subtype(binder.fold_with(folder)),
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
            ty::Predicate::Subtype(ref binder) => binder.visit_with(visitor),
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
            ty: self.ty.fold_with(folder),
            substs: self.substs.fold_with(folder),
            item_def_id: self.item_def_id,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor) || self.ty.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionTy<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::ProjectionTy {
            substs: self.substs.fold_with(folder),
            item_def_id: self.item_def_id,
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.substs.visit_with(visitor)
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
        ty::EquatePredicate(self.0.fold_with(folder), self.1.fold_with(folder))
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.0.visit_with(visitor) || self.1.visit_with(visitor)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::SubtypePredicate<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        ty::SubtypePredicate {
            a_is_expected: self.a_is_expected,
            a: self.a.fold_with(folder),
            b: self.b.fold_with(folder)
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.a.visit_with(visitor) || self.b.visit_with(visitor)
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

impl<'tcx> TypeFoldable<'tcx> for type_variable::Default<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        type_variable::Default {
            ty: self.ty.fold_with(folder),
            origin_span: self.origin_span,
            def_id: self.def_id
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.ty.visit_with(visitor)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>, I: Idx> TypeFoldable<'tcx> for IndexVec<I, T> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        self.iter().map(|x| x.fold_with(folder)).collect()
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.iter().any(|t| t.visit_with(visitor))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::error::TypeError<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        use ty::error::TypeError::*;

        match *self {
            Mismatch => Mismatch,
            UnsafetyMismatch(x) => UnsafetyMismatch(x.fold_with(folder)),
            AbiMismatch(x) => AbiMismatch(x.fold_with(folder)),
            Mutability => Mutability,
            TupleSize(x) => TupleSize(x),
            FixedArraySize(x) => FixedArraySize(x),
            ArgCount => ArgCount,
            RegionsDoesNotOutlive(a, b) => {
                RegionsDoesNotOutlive(a.fold_with(folder), b.fold_with(folder))
            },
            RegionsInsufficientlyPolymorphic(a, b) => {
                RegionsInsufficientlyPolymorphic(a, b.fold_with(folder))
            },
            RegionsOverlyPolymorphic(a, b) => {
                RegionsOverlyPolymorphic(a, b.fold_with(folder))
            },
            IntMismatch(x) => IntMismatch(x),
            FloatMismatch(x) => FloatMismatch(x),
            Traits(x) => Traits(x),
            VariadicMismatch(x) => VariadicMismatch(x),
            CyclicTy => CyclicTy,
            ProjectionMismatched(x) => ProjectionMismatched(x),
            ProjectionBoundsLength(x) => ProjectionBoundsLength(x),
            Sorts(x) => Sorts(x.fold_with(folder)),
            TyParamDefaultMismatch(ref x) => TyParamDefaultMismatch(x.fold_with(folder)),
            ExistentialMismatch(x) => ExistentialMismatch(x.fold_with(folder)),
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        use ty::error::TypeError::*;

        match *self {
            UnsafetyMismatch(x) => x.visit_with(visitor),
            AbiMismatch(x) => x.visit_with(visitor),
            RegionsDoesNotOutlive(a, b) => {
                a.visit_with(visitor) || b.visit_with(visitor)
            },
            RegionsInsufficientlyPolymorphic(_, b) |
            RegionsOverlyPolymorphic(_, b) => {
                b.visit_with(visitor)
            },
            Sorts(x) => x.visit_with(visitor),
            TyParamDefaultMismatch(ref x) => x.visit_with(visitor),
            ExistentialMismatch(x) => x.visit_with(visitor),
            Mismatch |
            Mutability |
            TupleSize(_) |
            FixedArraySize(_) |
            ArgCount |
            IntMismatch(_) |
            FloatMismatch(_) |
            Traits(_) |
            VariadicMismatch(_) |
            CyclicTy |
            ProjectionMismatched(_) |
            ProjectionBoundsLength(_) => false,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ConstVal<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ConstVal::Integral(i) => ConstVal::Integral(i),
            ConstVal::Float(f) => ConstVal::Float(f),
            ConstVal::Str(s) => ConstVal::Str(s),
            ConstVal::ByteStr(b) => ConstVal::ByteStr(b),
            ConstVal::Bool(b) => ConstVal::Bool(b),
            ConstVal::Char(c) => ConstVal::Char(c),
            ConstVal::Variant(def_id) => ConstVal::Variant(def_id),
            ConstVal::Function(def_id, substs) => {
                ConstVal::Function(def_id, substs.fold_with(folder))
            }
            ConstVal::Aggregate(ConstAggregate::Struct(fields)) => {
                let new_fields: Vec<_> = fields.iter().map(|&(name, v)| {
                    (name, v.fold_with(folder))
                }).collect();
                let fields = if new_fields == fields {
                    fields
                } else {
                    folder.tcx().alloc_name_const_slice(&new_fields)
                };
                ConstVal::Aggregate(ConstAggregate::Struct(fields))
            }
            ConstVal::Aggregate(ConstAggregate::Tuple(fields)) => {
                let new_fields: Vec<_> = fields.iter().map(|v| {
                    v.fold_with(folder)
                }).collect();
                let fields = if new_fields == fields {
                    fields
                } else {
                    folder.tcx().alloc_const_slice(&new_fields)
                };
                ConstVal::Aggregate(ConstAggregate::Tuple(fields))
            }
            ConstVal::Aggregate(ConstAggregate::Array(fields)) => {
                let new_fields: Vec<_> = fields.iter().map(|v| {
                    v.fold_with(folder)
                }).collect();
                let fields = if new_fields == fields {
                    fields
                } else {
                    folder.tcx().alloc_const_slice(&new_fields)
                };
                ConstVal::Aggregate(ConstAggregate::Array(fields))
            }
            ConstVal::Aggregate(ConstAggregate::Repeat(v, count)) => {
                let v = v.fold_with(folder);
                ConstVal::Aggregate(ConstAggregate::Repeat(v, count))
            }
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ConstVal::Integral(_) |
            ConstVal::Float(_) |
            ConstVal::Str(_) |
            ConstVal::ByteStr(_) |
            ConstVal::Bool(_) |
            ConstVal::Char(_) |
            ConstVal::Variant(_) => false,
            ConstVal::Function(_, substs) => substs.visit_with(visitor),
            ConstVal::Aggregate(ConstAggregate::Struct(fields)) => {
                fields.iter().any(|&(_, v)| v.visit_with(visitor))
            }
            ConstVal::Aggregate(ConstAggregate::Tuple(fields)) |
            ConstVal::Aggregate(ConstAggregate::Array(fields)) => {
                fields.iter().any(|v| v.visit_with(visitor))
            }
            ConstVal::Aggregate(ConstAggregate::Repeat(v, _)) => {
                v.visit_with(visitor)
            }
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for &'tcx ty::Const<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        let ty = self.ty.fold_with(folder);
        let val = self.val.fold_with(folder);
        folder.tcx().mk_const(ty::Const {
            ty,
            val
        })
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.ty.visit_with(visitor) || self.val.visit_with(visitor)
    }
}
