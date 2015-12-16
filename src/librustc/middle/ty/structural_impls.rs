// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst::{self, VecPerParamSpace};
use middle::traits;
use middle::ty::{self, TraitRef, Ty, TypeAndMut};
use middle::ty::{HasTypeFlags, Lift, TypeFlags, RegionEscape};
use middle::ty::fold::{TypeFoldable, TypeFolder};

use std::rc::Rc;
use syntax::abi;
use syntax::ptr::P;

use rustc_front::hir;

// FIXME(#20298) -- all of these traits basically walk various
// structures to test whether types/regions are reachable with various
// properties. It should be possible to express them in terms of one
// common "walker" trait or something.

impl<'tcx> RegionEscape for Ty<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.region_depth > depth
    }
}

impl<'tcx> RegionEscape for ty::TraitTy<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.principal.has_regions_escaping_depth(depth) ||
            self.bounds.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::ExistentialBounds<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.region_bound.has_regions_escaping_depth(depth) ||
            self.projection_bounds.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::InstantiatedPredicates<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.predicates.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for subst::Substs<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.types.has_regions_escaping_depth(depth) ||
            self.regions.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::ClosureSubsts<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.func_substs.has_regions_escaping_depth(depth) ||
            self.upvar_tys.iter().any(|t| t.has_regions_escaping_depth(depth))
    }
}

impl<T:RegionEscape> RegionEscape for Vec<T> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.iter().any(|t| t.has_regions_escaping_depth(depth))
    }
}

impl<'tcx> RegionEscape for ty::FnSig<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.inputs.has_regions_escaping_depth(depth) ||
            self.output.has_regions_escaping_depth(depth)
    }
}

impl<'tcx,T:RegionEscape> RegionEscape for VecPerParamSpace<T> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.iter_enumerated().any(|(space, _, t)| {
            if space == subst::FnSpace {
                t.has_regions_escaping_depth(depth+1)
            } else {
                t.has_regions_escaping_depth(depth)
            }
        })
    }
}

impl<'tcx> RegionEscape for ty::TypeScheme<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.ty.has_regions_escaping_depth(depth)
    }
}

impl RegionEscape for ty::Region {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.escapes_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::GenericPredicates<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.predicates.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::Predicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        match *self {
            ty::Predicate::Trait(ref data) => data.has_regions_escaping_depth(depth),
            ty::Predicate::Equate(ref data) => data.has_regions_escaping_depth(depth),
            ty::Predicate::RegionOutlives(ref data) => data.has_regions_escaping_depth(depth),
            ty::Predicate::TypeOutlives(ref data) => data.has_regions_escaping_depth(depth),
            ty::Predicate::Projection(ref data) => data.has_regions_escaping_depth(depth),
            ty::Predicate::WellFormed(ty) => ty.has_regions_escaping_depth(depth),
            ty::Predicate::ObjectSafe(_trait_def_id) => false,
        }
    }
}

impl<'tcx> RegionEscape for TraitRef<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.substs.types.iter().any(|t| t.has_regions_escaping_depth(depth)) ||
            self.substs.regions.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for subst::RegionSubsts {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        match *self {
            subst::ErasedRegions => false,
            subst::NonerasedRegions(ref r) => {
                r.iter().any(|t| t.has_regions_escaping_depth(depth))
            }
        }
    }
}

impl<'tcx,T:RegionEscape> RegionEscape for ty::Binder<T> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.0.has_regions_escaping_depth(depth + 1)
    }
}

impl<'tcx> RegionEscape for ty::FnOutput<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        match *self {
            ty::FnConverging(t) => t.has_regions_escaping_depth(depth),
            ty::FnDiverging => false
        }
    }
}

impl<'tcx> RegionEscape for ty::EquatePredicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.0.has_regions_escaping_depth(depth) || self.1.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::TraitPredicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.trait_ref.has_regions_escaping_depth(depth)
    }
}

impl<T:RegionEscape,U:RegionEscape> RegionEscape for ty::OutlivesPredicate<T,U> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.0.has_regions_escaping_depth(depth) || self.1.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::ProjectionPredicate<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.projection_ty.has_regions_escaping_depth(depth) ||
            self.ty.has_regions_escaping_depth(depth)
    }
}

impl<'tcx> RegionEscape for ty::ProjectionTy<'tcx> {
    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.trait_ref.has_regions_escaping_depth(depth)
    }
}

impl HasTypeFlags for () {
    fn has_type_flags(&self, _flags: TypeFlags) -> bool {
        false
    }
}

impl<'tcx,T:HasTypeFlags> HasTypeFlags for Vec<T> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self[..].has_type_flags(flags)
    }
}

impl<'tcx,T:HasTypeFlags> HasTypeFlags for [T] {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.iter().any(|p| p.has_type_flags(flags))
    }
}

impl<'tcx,T:HasTypeFlags> HasTypeFlags for VecPerParamSpace<T> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.iter().any(|p| p.has_type_flags(flags))
    }
}

impl HasTypeFlags for abi::Abi {
    fn has_type_flags(&self, _flags: TypeFlags) -> bool {
        false
    }
}

impl HasTypeFlags for hir::Unsafety {
    fn has_type_flags(&self, _flags: TypeFlags) -> bool {
        false
    }
}

impl HasTypeFlags for ty::BuiltinBounds {
    fn has_type_flags(&self, _flags: TypeFlags) -> bool {
        false
    }
}

impl<'tcx> HasTypeFlags for ty::ClosureTy<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.sig.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::ClosureUpvar<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.ty.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::ExistentialBounds<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.projection_bounds.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::InstantiatedPredicates<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.predicates.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::Predicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        match *self {
            ty::Predicate::Trait(ref data) => data.has_type_flags(flags),
            ty::Predicate::Equate(ref data) => data.has_type_flags(flags),
            ty::Predicate::RegionOutlives(ref data) => data.has_type_flags(flags),
            ty::Predicate::TypeOutlives(ref data) => data.has_type_flags(flags),
            ty::Predicate::Projection(ref data) => data.has_type_flags(flags),
            ty::Predicate::WellFormed(data) => data.has_type_flags(flags),
            ty::Predicate::ObjectSafe(_trait_def_id) => false,
        }
    }
}

impl<'tcx> HasTypeFlags for ty::TraitPredicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.trait_ref.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::EquatePredicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.0.has_type_flags(flags) || self.1.has_type_flags(flags)
    }
}

impl HasTypeFlags for ty::Region {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        if flags.intersects(TypeFlags::HAS_LOCAL_NAMES) {
            // does this represent a region that cannot be named in a global
            // way? used in fulfillment caching.
            match *self {
                ty::ReStatic | ty::ReEmpty => {}
                _ => return true
            }
        }
        if flags.intersects(TypeFlags::HAS_RE_INFER) {
            match *self {
                ty::ReVar(_) | ty::ReSkolemized(..) => { return true }
                _ => {}
            }
        }
        false
    }
}

impl<T:HasTypeFlags,U:HasTypeFlags> HasTypeFlags for ty::OutlivesPredicate<T,U> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.0.has_type_flags(flags) || self.1.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::ProjectionPredicate<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.projection_ty.has_type_flags(flags) || self.ty.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::ProjectionTy<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.trait_ref.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for Ty<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.flags.get().intersects(flags)
    }
}

impl<'tcx> HasTypeFlags for TypeAndMut<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.ty.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for TraitRef<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.substs.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for subst::Substs<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.types.has_type_flags(flags) || match self.regions {
            subst::ErasedRegions => false,
            subst::NonerasedRegions(ref r) => r.has_type_flags(flags)
        }
    }
}

impl<'tcx,T> HasTypeFlags for Option<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.iter().any(|t| t.has_type_flags(flags))
    }
}

impl<'tcx,T> HasTypeFlags for Rc<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        (**self).has_type_flags(flags)
    }
}

impl<'tcx,T> HasTypeFlags for Box<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        (**self).has_type_flags(flags)
    }
}

impl<T> HasTypeFlags for ty::Binder<T>
    where T : HasTypeFlags
{
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.0.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::FnOutput<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        match *self {
            ty::FnConverging(t) => t.has_type_flags(flags),
            ty::FnDiverging => false,
        }
    }
}

impl<'tcx> HasTypeFlags for ty::FnSig<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.inputs.iter().any(|t| t.has_type_flags(flags)) ||
            self.output.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::BareFnTy<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.sig.has_type_flags(flags)
    }
}

impl<'tcx> HasTypeFlags for ty::ClosureSubsts<'tcx> {
    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.func_substs.has_type_flags(flags) ||
            self.upvar_tys.iter().any(|t| t.has_type_flags(flags))
    }
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

impl<'tcx, A: Lift<'tcx>, B: Lift<'tcx>> Lift<'tcx> for (A, B) {
    type Lifted = (A::Lifted, B::Lifted);
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&self.0).and_then(|a| tcx.lift(&self.1).map(|b| (a, b)))
    }
}

impl<'tcx, T: Lift<'tcx>> Lift<'tcx> for [T] {
    type Lifted = Vec<T::Lifted>;
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<Self::Lifted> {
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
    fn lift_to_tcx(&self, _: &ty::ctxt<'tcx>) -> Option<ty::Region> {
        Some(*self)
    }
}

impl<'a, 'tcx> Lift<'tcx> for TraitRef<'a> {
    type Lifted = TraitRef<'tcx>;
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<TraitRef<'tcx>> {
        tcx.lift(&self.substs).map(|substs| TraitRef {
            def_id: self.def_id,
            substs: substs
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::TraitPredicate<'a> {
    type Lifted = ty::TraitPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<ty::TraitPredicate<'tcx>> {
        tcx.lift(&self.trait_ref).map(|trait_ref| ty::TraitPredicate {
            trait_ref: trait_ref
        })
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::EquatePredicate<'a> {
    type Lifted = ty::EquatePredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<ty::EquatePredicate<'tcx>> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| ty::EquatePredicate(a, b))
    }
}

impl<'tcx, A: Copy+Lift<'tcx>, B: Copy+Lift<'tcx>> Lift<'tcx> for ty::OutlivesPredicate<A, B> {
    type Lifted = ty::OutlivesPredicate<A::Lifted, B::Lifted>;
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<Self::Lifted> {
        tcx.lift(&(self.0, self.1)).map(|(a, b)| ty::OutlivesPredicate(a, b))
    }
}

impl<'a, 'tcx> Lift<'tcx> for ty::ProjectionPredicate<'a> {
    type Lifted = ty::ProjectionPredicate<'tcx>;
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<ty::ProjectionPredicate<'tcx>> {
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
    fn lift_to_tcx(&self, tcx: &ty::ctxt<'tcx>) -> Option<Self::Lifted> {
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
                fn fold_with<F:TypeFolder<'tcx>>(&self, _: &mut F) -> $ty {
                    *self
                }
            }
        )+
    }
}

CopyImpls! { (), hir::Unsafety, abi::Abi }

impl<'tcx, T:TypeFoldable<'tcx>, U:TypeFoldable<'tcx>> TypeFoldable<'tcx> for (T, U) {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> (T, U) {
        (self.0.fold_with(folder), self.1.fold_with(folder))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Option<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Option<T> {
        self.as_ref().map(|t| t.fold_with(folder))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Rc<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Rc<T> {
        Rc::new((**self).fold_with(folder))
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Box<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Box<T> {
        let content: T = (**self).fold_with(folder);
        box content
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for Vec<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Vec<T> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<'tcx, T:TypeFoldable<'tcx>> TypeFoldable<'tcx> for ty::Binder<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Binder<T> {
        folder.fold_binder(self)
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for P<[T]> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> P<[T]> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for VecPerParamSpace<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> VecPerParamSpace<T> {

        // Things in the Fn space take place under an additional level
        // of region binding relative to the other spaces. This is
        // because those entries are attached to a method, and methods
        // always introduce a level of region binding.

        let result = self.map_enumerated(|(space, index, elem)| {
            if space == subst::FnSpace && index == 0 {
                // enter new level when/if we reach the first thing in fn space
                folder.enter_region_binder();
            }
            elem.fold_with(folder)
        });
        if result.len(subst::FnSpace) > 0 {
            // if there was anything in fn space, exit the region binding level
            folder.exit_region_binder();
        }
        result
    }
}

impl<'tcx> TypeFoldable<'tcx> for Ty<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Ty<'tcx> {
        folder.fold_ty(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::BareFnTy<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::BareFnTy<'tcx> {
        folder.fold_bare_fn_ty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureTy<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ClosureTy<'tcx> {
        folder.fold_closure_ty(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeAndMut<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TypeAndMut<'tcx> {
        folder.fold_mt(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnOutput<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::FnOutput<'tcx> {
        folder.fold_output(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::FnSig<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::FnSig<'tcx> {
        folder.fold_fn_sig(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitRef<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TraitRef<'tcx> {
        folder.fold_trait_ref(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Region {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Region {
        folder.fold_region(*self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for subst::Substs<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> subst::Substs<'tcx> {
        folder.fold_substs(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureSubsts<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ClosureSubsts<'tcx> {
        let func_substs = self.func_substs.fold_with(folder);
        ty::ClosureSubsts {
            func_substs: folder.tcx().mk_substs(func_substs),
            upvar_tys: self.upvar_tys.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ItemSubsts<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ItemSubsts<'tcx> {
        ty::ItemSubsts {
            substs: self.substs.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::adjustment::AutoRef<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::adjustment::AutoRef<'tcx> {
        folder.fold_autoref(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::BuiltinBounds {
    fn fold_with<F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> ty::BuiltinBounds {
        *self
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ExistentialBounds<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ExistentialBounds<'tcx> {
        folder.fold_existential_bounds(self)
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TypeParameterDef<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TypeParameterDef<'tcx> {
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
}

impl<'tcx> TypeFoldable<'tcx> for ty::ObjectLifetimeDefault {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ObjectLifetimeDefault {
        match *self {
            ty::ObjectLifetimeDefault::Ambiguous =>
                ty::ObjectLifetimeDefault::Ambiguous,

            ty::ObjectLifetimeDefault::BaseDefault =>
                ty::ObjectLifetimeDefault::BaseDefault,

            ty::ObjectLifetimeDefault::Specific(r) =>
                ty::ObjectLifetimeDefault::Specific(r.fold_with(folder)),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::RegionParameterDef {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::RegionParameterDef {
        ty::RegionParameterDef {
            name: self.name,
            def_id: self.def_id,
            space: self.space,
            index: self.index,
            bounds: self.bounds.fold_with(folder)
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Generics<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Generics<'tcx> {
        ty::Generics {
            types: self.types.fold_with(folder),
            regions: self.regions.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::GenericPredicates<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::GenericPredicates<'tcx> {
        ty::GenericPredicates {
            predicates: self.predicates.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::Predicate<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Predicate<'tcx> {
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
            ty::Predicate::ObjectSafe(trait_def_id) =>
                ty::Predicate::ObjectSafe(trait_def_id),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionPredicate<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ProjectionPredicate<'tcx> {
        ty::ProjectionPredicate {
            projection_ty: self.projection_ty.fold_with(folder),
            ty: self.ty.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ProjectionTy<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ProjectionTy<'tcx> {
        ty::ProjectionTy {
            trait_ref: self.trait_ref.fold_with(folder),
            item_name: self.item_name,
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::InstantiatedPredicates<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::InstantiatedPredicates<'tcx> {
        ty::InstantiatedPredicates {
            predicates: self.predicates.fold_with(folder),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::EquatePredicate<'tcx> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::EquatePredicate<'tcx> {
        ty::EquatePredicate(self.0.fold_with(folder),
                            self.1.fold_with(folder))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::TraitPredicate<'tcx> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TraitPredicate<'tcx> {
        ty::TraitPredicate {
            trait_ref: self.trait_ref.fold_with(folder)
        }
    }
}

impl<'tcx,T,U> TypeFoldable<'tcx> for ty::OutlivesPredicate<T,U>
    where T : TypeFoldable<'tcx>,
          U : TypeFoldable<'tcx>,
{
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::OutlivesPredicate<T,U> {
        ty::OutlivesPredicate(self.0.fold_with(folder),
                              self.1.fold_with(folder))
    }
}

impl<'tcx> TypeFoldable<'tcx> for ty::ClosureUpvar<'tcx> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ClosureUpvar<'tcx> {
        ty::ClosureUpvar {
            def: self.def,
            span: self.span,
            ty: self.ty.fold_with(folder),
        }
    }
}

impl<'a, 'tcx> TypeFoldable<'tcx> for ty::ParameterEnvironment<'a, 'tcx> where 'tcx: 'a {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ParameterEnvironment<'a, 'tcx> {
        ty::ParameterEnvironment {
            tcx: self.tcx,
            free_substs: self.free_substs.fold_with(folder),
            implicit_region_bound: self.implicit_region_bound.fold_with(folder),
            caller_bounds: self.caller_bounds.fold_with(folder),
            selection_cache: traits::SelectionCache::new(),
            evaluation_cache: traits::EvaluationCache::new(),
            free_id: self.free_id,
        }
    }
}
