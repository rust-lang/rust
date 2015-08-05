// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generalized type folding mechanism. The setup is a bit convoluted
//! but allows for convenient usage. Let T be an instance of some
//! "foldable type" (one which implements `TypeFoldable`) and F be an
//! instance of a "folder" (a type which implements `TypeFolder`). Then
//! the setup is intended to be:
//!
//!     T.fold_with(F) --calls--> F.fold_T(T) --calls--> super_fold_T(F, T)
//!
//! This way, when you define a new folder F, you can override
//! `fold_T()` to customize the behavior, and invoke `super_fold_T()`
//! to get the original behavior. Meanwhile, to actually fold
//! something, you can just write `T.fold_with(F)`, which is
//! convenient. (Note that `fold_with` will also transparently handle
//! things like a `Vec<T>` where T is foldable and so on.)
//!
//! In this ideal setup, the only function that actually *does*
//! anything is `super_fold_T`, which traverses the type `T`. Moreover,
//! `super_fold_T` should only ever call `T.fold_with()`.
//!
//! In some cases, we follow a degenerate pattern where we do not have
//! a `fold_T` nor `super_fold_T` method. Instead, `T.fold_with`
//! traverses the structure directly. This is suboptimal because the
//! behavior cannot be overridden, but it's much less work to implement.
//! If you ever *do* need an override that doesn't exist, it's not hard
//! to convert the degenerate pattern into the proper thing.

use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty::{self, Ty, HasTypeFlags, RegionEscape};
use middle::traits;

use std::fmt;
use std::rc::Rc;
use syntax::abi;
use syntax::ast;
use syntax::owned_slice::OwnedSlice;
use util::nodemap::{FnvHashMap, FnvHashSet};

///////////////////////////////////////////////////////////////////////////
// Two generic traits

/// The TypeFoldable trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in TypeFolder.
pub trait TypeFoldable<'tcx>: fmt::Debug + Clone {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self;
}

/// The TypeFolder trait defines the actual *folding*. There is a
/// method defined for every foldable type. Each of these has a
/// default implementation that does an "identity" fold. Within each
/// identity fold, it should invoke `foo.fold_with(self)` to fold each
/// sub-item.
pub trait TypeFolder<'tcx> : Sized {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx>;

    /// Invoked by the `super_*` routines when we enter a region
    /// binding level (for example, when entering a function
    /// signature). This is used by clients that want to track the
    /// Debruijn index nesting level.
    fn enter_region_binder(&mut self) { }

    /// Invoked by the `super_*` routines when we exit a region
    /// binding level. This is used by clients that want to
    /// track the Debruijn index nesting level.
    fn exit_region_binder(&mut self) { }

    fn fold_binder<T>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T>
        where T : TypeFoldable<'tcx>
    {
        // FIXME(#20526) this should replace `enter_region_binder`/`exit_region_binder`.
        super_fold_binder(self, t)
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        super_fold_ty(self, t)
    }

    fn fold_mt(&mut self, t: &ty::TypeAndMut<'tcx>) -> ty::TypeAndMut<'tcx> {
        super_fold_mt(self, t)
    }

    fn fold_trait_ref(&mut self, t: &ty::TraitRef<'tcx>) -> ty::TraitRef<'tcx> {
        super_fold_trait_ref(self, t)
    }

    fn fold_substs(&mut self,
                   substs: &subst::Substs<'tcx>)
                   -> subst::Substs<'tcx> {
        super_fold_substs(self, substs)
    }

    fn fold_fn_sig(&mut self,
                   sig: &ty::FnSig<'tcx>)
                   -> ty::FnSig<'tcx> {
        super_fold_fn_sig(self, sig)
    }

    fn fold_output(&mut self,
                      output: &ty::FnOutput<'tcx>)
                      -> ty::FnOutput<'tcx> {
        super_fold_output(self, output)
    }

    fn fold_bare_fn_ty(&mut self,
                       fty: &ty::BareFnTy<'tcx>)
                       -> ty::BareFnTy<'tcx>
    {
        super_fold_bare_fn_ty(self, fty)
    }

    fn fold_closure_ty(&mut self,
                       fty: &ty::ClosureTy<'tcx>)
                       -> ty::ClosureTy<'tcx> {
        super_fold_closure_ty(self, fty)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        r
    }

    fn fold_existential_bounds(&mut self, s: &ty::ExistentialBounds<'tcx>)
                               -> ty::ExistentialBounds<'tcx> {
        super_fold_existential_bounds(self, s)
    }

    fn fold_autoref(&mut self, ar: &ty::AutoRef<'tcx>) -> ty::AutoRef<'tcx> {
        super_fold_autoref(self, ar)
    }

    fn fold_item_substs(&mut self, i: ty::ItemSubsts<'tcx>) -> ty::ItemSubsts<'tcx> {
        super_fold_item_substs(self, i)
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

CopyImpls! { (), ast::Unsafety, abi::Abi }

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

impl<'tcx, T: TypeFoldable<'tcx>> TypeFoldable<'tcx> for OwnedSlice<T> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> OwnedSlice<T> {
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

impl<'tcx> TypeFoldable<'tcx> for ty::AutoRef<'tcx> {
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::AutoRef<'tcx> {
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

impl<'tcx,O> TypeFoldable<'tcx> for traits::Obligation<'tcx,O>
    where O : TypeFoldable<'tcx>
{
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::Obligation<'tcx, O> {
        traits::Obligation {
            cause: self.cause.clone(),
            recursion_depth: self.recursion_depth,
            predicate: self.predicate.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableImplData<'tcx, N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableImplData<'tcx, N> {
        traits::VtableImplData {
            impl_def_id: self.impl_def_id,
            substs: self.substs.fold_with(folder),
            nested: self.nested.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableClosureData<'tcx, N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableClosureData<'tcx, N> {
        traits::VtableClosureData {
            closure_def_id: self.closure_def_id,
            substs: self.substs.fold_with(folder),
            nested: self.nested.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableDefaultImplData<N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableDefaultImplData<N> {
        traits::VtableDefaultImplData {
            trait_def_id: self.trait_def_id,
            nested: self.nested.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::VtableBuiltinData<N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableBuiltinData<N> {
        traits::VtableBuiltinData {
            nested: self.nested.fold_with(folder),
        }
    }
}

impl<'tcx, N: TypeFoldable<'tcx>> TypeFoldable<'tcx> for traits::Vtable<'tcx, N> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::Vtable<'tcx, N> {
        match *self {
            traits::VtableImpl(ref v) => traits::VtableImpl(v.fold_with(folder)),
            traits::VtableDefaultImpl(ref t) => traits::VtableDefaultImpl(t.fold_with(folder)),
            traits::VtableClosure(ref d) => {
                traits::VtableClosure(d.fold_with(folder))
            }
            traits::VtableFnPointer(ref d) => {
                traits::VtableFnPointer(d.fold_with(folder))
            }
            traits::VtableParam(ref n) => traits::VtableParam(n.fold_with(folder)),
            traits::VtableBuiltin(ref d) => traits::VtableBuiltin(d.fold_with(folder)),
            traits::VtableObject(ref d) => traits::VtableObject(d.fold_with(folder)),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for traits::VtableObjectData<'tcx> {
    fn fold_with<F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableObjectData<'tcx> {
        traits::VtableObjectData {
            upcast_trait_ref: self.upcast_trait_ref.fold_with(folder),
            vtable_base: self.vtable_base
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
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// "super" routines: these are the default implementations for TypeFolder.
//
// They should invoke `foo.fold_with()` to do recursive folding.

pub fn super_fold_binder<'tcx, T, U>(this: &mut T,
                                     binder: &ty::Binder<U>)
                                     -> ty::Binder<U>
    where T : TypeFolder<'tcx>, U : TypeFoldable<'tcx>
{
    this.enter_region_binder();
    let result = ty::Binder(binder.0.fold_with(this));
    this.exit_region_binder();
    result
}

pub fn super_fold_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                ty: Ty<'tcx>)
                                                -> Ty<'tcx> {
    let sty = match ty.sty {
        ty::TyBox(typ) => {
            ty::TyBox(typ.fold_with(this))
        }
        ty::TyRawPtr(ref tm) => {
            ty::TyRawPtr(tm.fold_with(this))
        }
        ty::TyArray(typ, sz) => {
            ty::TyArray(typ.fold_with(this), sz)
        }
        ty::TySlice(typ) => {
            ty::TySlice(typ.fold_with(this))
        }
        ty::TyEnum(tid, ref substs) => {
            let substs = substs.fold_with(this);
            ty::TyEnum(tid, this.tcx().mk_substs(substs))
        }
        ty::TyTrait(box ty::TraitTy { ref principal, ref bounds }) => {
            ty::TyTrait(box ty::TraitTy {
                principal: principal.fold_with(this),
                bounds: bounds.fold_with(this),
            })
        }
        ty::TyTuple(ref ts) => {
            ty::TyTuple(ts.fold_with(this))
        }
        ty::TyBareFn(opt_def_id, ref f) => {
            let bfn = f.fold_with(this);
            ty::TyBareFn(opt_def_id, this.tcx().mk_bare_fn(bfn))
        }
        ty::TyRef(r, ref tm) => {
            let r = r.fold_with(this);
            ty::TyRef(this.tcx().mk_region(r), tm.fold_with(this))
        }
        ty::TyStruct(did, ref substs) => {
            let substs = substs.fold_with(this);
            ty::TyStruct(did, this.tcx().mk_substs(substs))
        }
        ty::TyClosure(did, ref substs) => {
            let s = substs.fold_with(this);
            ty::TyClosure(did, s)
        }
        ty::TyProjection(ref data) => {
            ty::TyProjection(data.fold_with(this))
        }
        ty::TyBool | ty::TyChar | ty::TyStr |
        ty::TyInt(_) | ty::TyUint(_) | ty::TyFloat(_) |
        ty::TyError | ty::TyInfer(_) |
        ty::TyParam(..) => {
            ty.sty.clone()
        }
    };
    this.tcx().mk_ty(sty)
}

pub fn super_fold_substs<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                    substs: &subst::Substs<'tcx>)
                                                    -> subst::Substs<'tcx> {
    let regions = match substs.regions {
        subst::ErasedRegions => {
            subst::ErasedRegions
        }
        subst::NonerasedRegions(ref regions) => {
            subst::NonerasedRegions(regions.fold_with(this))
        }
    };

    subst::Substs { regions: regions,
                    types: substs.types.fold_with(this) }
}

pub fn super_fold_fn_sig<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                    sig: &ty::FnSig<'tcx>)
                                                    -> ty::FnSig<'tcx>
{
    ty::FnSig { inputs: sig.inputs.fold_with(this),
                output: sig.output.fold_with(this),
                variadic: sig.variadic }
}

pub fn super_fold_output<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                    output: &ty::FnOutput<'tcx>)
                                                    -> ty::FnOutput<'tcx> {
    match *output {
        ty::FnConverging(ref ty) => ty::FnConverging(ty.fold_with(this)),
        ty::FnDiverging => ty::FnDiverging
    }
}

pub fn super_fold_bare_fn_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                        fty: &ty::BareFnTy<'tcx>)
                                                        -> ty::BareFnTy<'tcx>
{
    ty::BareFnTy { sig: fty.sig.fold_with(this),
                   abi: fty.abi,
                   unsafety: fty.unsafety }
}

pub fn super_fold_closure_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                        fty: &ty::ClosureTy<'tcx>)
                                                        -> ty::ClosureTy<'tcx>
{
    ty::ClosureTy {
        sig: fty.sig.fold_with(this),
        unsafety: fty.unsafety,
        abi: fty.abi,
    }
}

pub fn super_fold_trait_ref<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                       t: &ty::TraitRef<'tcx>)
                                                       -> ty::TraitRef<'tcx>
{
    let substs = t.substs.fold_with(this);
    ty::TraitRef {
        def_id: t.def_id,
        substs: this.tcx().mk_substs(substs),
    }
}

pub fn super_fold_mt<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                mt: &ty::TypeAndMut<'tcx>)
                                                -> ty::TypeAndMut<'tcx> {
    ty::TypeAndMut {ty: mt.ty.fold_with(this),
            mutbl: mt.mutbl}
}

pub fn super_fold_existential_bounds<'tcx, T: TypeFolder<'tcx>>(
    this: &mut T,
    bounds: &ty::ExistentialBounds<'tcx>)
    -> ty::ExistentialBounds<'tcx>
{
    ty::ExistentialBounds {
        region_bound: bounds.region_bound.fold_with(this),
        builtin_bounds: bounds.builtin_bounds,
        projection_bounds: bounds.projection_bounds.fold_with(this),
    }
}

pub fn super_fold_autoref<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                     autoref: &ty::AutoRef<'tcx>)
                                                     -> ty::AutoRef<'tcx>
{
    match *autoref {
        ty::AutoPtr(r, m) => {
            let r = r.fold_with(this);
            ty::AutoPtr(this.tcx().mk_region(r), m)
        }
        ty::AutoUnsafe(m) => ty::AutoUnsafe(m)
    }
}

pub fn super_fold_item_substs<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                         substs: ty::ItemSubsts<'tcx>)
                                                         -> ty::ItemSubsts<'tcx>
{
    ty::ItemSubsts {
        substs: substs.substs.fold_with(this),
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'a, 'tcx: 'a, F> where F: FnMut(Ty<'tcx>) -> Ty<'tcx> {
    pub tcx: &'a ty::ctxt<'tcx>,
    pub fldop: F,
}

impl<'a, 'tcx, F> TypeFolder<'tcx> for BottomUpFolder<'a, 'tcx, F> where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
{
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.tcx }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t1 = super_fold_ty(self, ty);
        (self.fldop)(t1)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

/// Folds over the substructure of a type, visiting its component
/// types and all regions that occur *free* within it.
///
/// That is, `Ty` can contain function or method types that bind
/// regions at the call site (`ReLateBound`), and occurrences of
/// regions (aka "lifetimes") that are bound within a type are not
/// visited by this folder; only regions that occur free will be
/// visited by `fld_r`.

pub struct RegionFolder<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    skipped_regions: &'a mut bool,
    current_depth: u32,
    fld_r: &'a mut (FnMut(ty::Region, u32) -> ty::Region + 'a),
}

impl<'a, 'tcx> RegionFolder<'a, 'tcx> {
    pub fn new<F>(tcx: &'a ty::ctxt<'tcx>,
                  skipped_regions: &'a mut bool,
                  fld_r: &'a mut F) -> RegionFolder<'a, 'tcx>
        where F : FnMut(ty::Region, u32) -> ty::Region
    {
        RegionFolder {
            tcx: tcx,
            skipped_regions: skipped_regions,
            current_depth: 1,
            fld_r: fld_r,
        }
    }
}

/// Collects the free and escaping regions in `value` into `region_set`. Returns
/// whether any late-bound regions were skipped
pub fn collect_regions<'tcx,T>(tcx: &ty::ctxt<'tcx>,
                               value: &T,
                               region_set: &mut FnvHashSet<ty::Region>) -> bool
    where T : TypeFoldable<'tcx>
{
    let mut have_bound_regions = false;
    fold_regions(tcx, value, &mut have_bound_regions,
                 |r, d| { region_set.insert(r.from_depth(d)); r });
    have_bound_regions
}

/// Folds the escaping and free regions in `value` using `f`, and
/// sets `skipped_regions` to true if any late-bound region was found
/// and skipped.
pub fn fold_regions<'tcx,T,F>(tcx: &ty::ctxt<'tcx>,
                              value: &T,
                              skipped_regions: &mut bool,
                              mut f: F)
                              -> T
    where F : FnMut(ty::Region, u32) -> ty::Region,
          T : TypeFoldable<'tcx>,
{
    value.fold_with(&mut RegionFolder::new(tcx, skipped_regions, &mut f))
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionFolder<'a, 'tcx>
{
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.tcx }

    fn enter_region_binder(&mut self) {
        self.current_depth += 1;
    }

    fn exit_region_binder(&mut self) {
        self.current_depth -= 1;
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReLateBound(debruijn, _) if debruijn.depth < self.current_depth => {
                debug!("RegionFolder.fold_region({:?}) skipped bound region (current depth={})",
                       r, self.current_depth);
                *self.skipped_regions = true;
                r
            }
            _ => {
                debug!("RegionFolder.fold_region({:?}) folding free region (current_depth={})",
                       r, self.current_depth);
                (self.fld_r)(r, self.current_depth)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Late-bound region replacer

// Replaces the escaping regions in a type.

struct RegionReplacer<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    current_depth: u32,
    fld_r: &'a mut (FnMut(ty::BoundRegion) -> ty::Region + 'a),
    map: FnvHashMap<ty::BoundRegion, ty::Region>
}

impl<'a, 'tcx> RegionReplacer<'a, 'tcx> {
    fn new<F>(tcx: &'a ty::ctxt<'tcx>, fld_r: &'a mut F) -> RegionReplacer<'a, 'tcx>
        where F : FnMut(ty::BoundRegion) -> ty::Region
    {
        RegionReplacer {
            tcx: tcx,
            current_depth: 1,
            fld_r: fld_r,
            map: FnvHashMap()
        }
    }
}

pub fn replace_late_bound_regions<'tcx,T,F>(tcx: &ty::ctxt<'tcx>,
                                            value: &ty::Binder<T>,
                                            mut f: F)
                                            -> (T, FnvHashMap<ty::BoundRegion, ty::Region>)
    where F : FnMut(ty::BoundRegion) -> ty::Region,
          T : TypeFoldable<'tcx>,
{
    debug!("replace_late_bound_regions({:?})", value);
    let mut replacer = RegionReplacer::new(tcx, &mut f);
    let result = value.skip_binder().fold_with(&mut replacer);
    (result, replacer.map)
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionReplacer<'a, 'tcx>
{
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.tcx }

    fn enter_region_binder(&mut self) {
        self.current_depth += 1;
    }

    fn exit_region_binder(&mut self) {
        self.current_depth -= 1;
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_regions_escaping_depth(self.current_depth-1) {
            return t;
        }

        super_fold_ty(self, t)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReLateBound(debruijn, br) if debruijn.depth == self.current_depth => {
                debug!("RegionReplacer.fold_region({:?}) folding region (current_depth={})",
                       r, self.current_depth);
                let fld_r = &mut self.fld_r;
                let region = *self.map.entry(br).or_insert_with(|| fld_r(br));
                if let ty::ReLateBound(debruijn1, br) = region {
                    // If the callback returns a late-bound region,
                    // that region should always use depth 1. Then we
                    // adjust it to the correct depth.
                    assert_eq!(debruijn1.depth, 1);
                    ty::ReLateBound(debruijn, br)
                } else {
                    region
                }
            }
            r => r
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Region eraser
//
// Replaces all free regions with 'static. Useful in contexts, such as
// method probing, where precise region relationships are not
// important. Note that in trans you should use
// `common::erase_regions` instead.

pub struct RegionEraser<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

pub fn erase_regions<'tcx, T: TypeFoldable<'tcx>>(tcx: &ty::ctxt<'tcx>, t: T) -> T {
    let mut eraser = RegionEraser { tcx: tcx };
    t.fold_with(&mut eraser)
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionEraser<'a, 'tcx> {
    fn tcx(&self) -> &ty::ctxt<'tcx> { self.tcx }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_erasable_regions() {
            return t;
        }

        super_fold_ty(self, t)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        // because whether or not a region is bound affects subtyping,
        // we can't erase the bound/free distinction, but we can
        // replace all free regions with 'static
        match r {
            ty::ReLateBound(..) | ty::ReEarlyBound(..) => r,
            _ => ty::ReStatic
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Region shifter
//
// Shifts the De Bruijn indices on all escaping bound regions by a
// fixed amount. Useful in substitution or when otherwise introducing
// a binding level that is not intended to capture the existing bound
// regions. See comment on `shift_regions_through_binders` method in
// `subst.rs` for more details.

pub fn shift_region(region: ty::Region, amount: u32) -> ty::Region {
    match region {
        ty::ReLateBound(debruijn, br) => {
            ty::ReLateBound(debruijn.shifted(amount), br)
        }
        _ => {
            region
        }
    }
}

pub fn shift_regions<'tcx, T:TypeFoldable<'tcx>>(tcx: &ty::ctxt<'tcx>,
                                                 amount: u32, value: &T) -> T {
    debug!("shift_regions(value={:?}, amount={})",
           value, amount);

    value.fold_with(&mut RegionFolder::new(tcx, &mut false, &mut |region, _current_depth| {
        shift_region(region, amount)
    }))
}
