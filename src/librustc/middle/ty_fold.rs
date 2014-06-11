// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Generalized type folding mechanism.

use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty;
use middle::typeck;
use std::rc::Rc;
use syntax::owned_slice::OwnedSlice;
use util::ppaux::Repr;

///////////////////////////////////////////////////////////////////////////
// Two generic traits

/// The TypeFoldable trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in TypeFolder.
pub trait TypeFoldable {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> Self;
}

/// The TypeFolder trait defines the actual *folding*. There is a
/// method defined for every foldable type. Each of these has a
/// default implementation that does an "identity" fold. Within each
/// identity fold, it should invoke `foo.fold_with(self)` to fold each
/// sub-item.
pub trait TypeFolder {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt;

    fn fold_ty(&mut self, t: ty::t) -> ty::t {
        super_fold_ty(self, t)
    }

    fn fold_mt(&mut self, t: &ty::mt) -> ty::mt {
        super_fold_mt(self, t)
    }

    fn fold_trait_ref(&mut self, t: &ty::TraitRef) -> ty::TraitRef {
        super_fold_trait_ref(self, t)
    }

    fn fold_sty(&mut self, sty: &ty::sty) -> ty::sty {
        super_fold_sty(self, sty)
    }

    fn fold_substs(&mut self,
                   substs: &subst::Substs)
                   -> subst::Substs {
        super_fold_substs(self, substs)
    }

    fn fold_sig(&mut self,
                sig: &ty::FnSig)
                -> ty::FnSig {
        super_fold_sig(self, sig)
    }

    fn fold_bare_fn_ty(&mut self,
                       fty: &ty::BareFnTy)
                       -> ty::BareFnTy
    {
        super_fold_bare_fn_ty(self, fty)
    }

    fn fold_closure_ty(&mut self,
                       fty: &ty::ClosureTy)
                       -> ty::ClosureTy {
        super_fold_closure_ty(self, fty)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        r
    }

    fn fold_trait_store(&mut self, s: ty::TraitStore) -> ty::TraitStore {
        super_fold_trait_store(self, s)
    }

    fn fold_autoref(&mut self, ar: &ty::AutoRef) -> ty::AutoRef {
        super_fold_autoref(self, ar)
    }

    fn fold_item_substs(&mut self, i: ty::ItemSubsts) -> ty::ItemSubsts {
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

impl<T:TypeFoldable> TypeFoldable for Option<T> {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> Option<T> {
        self.as_ref().map(|t| t.fold_with(folder))
    }
}

impl<T:TypeFoldable> TypeFoldable for Rc<T> {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> Rc<T> {
        Rc::new((**self).fold_with(folder))
    }
}

impl<T:TypeFoldable> TypeFoldable for Vec<T> {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> Vec<T> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<T:TypeFoldable> TypeFoldable for OwnedSlice<T> {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> OwnedSlice<T> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<T:TypeFoldable> TypeFoldable for VecPerParamSpace<T> {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> VecPerParamSpace<T> {
        self.map(|t| t.fold_with(folder))
    }
}

impl TypeFoldable for ty::TraitStore {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::TraitStore {
        folder.fold_trait_store(*self)
    }
}

impl TypeFoldable for ty::t {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::t {
        folder.fold_ty(*self)
    }
}

impl TypeFoldable for ty::BareFnTy {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::BareFnTy {
        folder.fold_bare_fn_ty(self)
    }
}

impl TypeFoldable for ty::ClosureTy {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::ClosureTy {
        folder.fold_closure_ty(self)
    }
}

impl TypeFoldable for ty::mt {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::mt {
        folder.fold_mt(self)
    }
}

impl TypeFoldable for ty::FnSig {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::FnSig {
        folder.fold_sig(self)
    }
}

impl TypeFoldable for ty::sty {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::sty {
        folder.fold_sty(self)
    }
}

impl TypeFoldable for ty::TraitRef {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::TraitRef {
        folder.fold_trait_ref(self)
    }
}

impl TypeFoldable for ty::Region {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::Region {
        folder.fold_region(*self)
    }
}

impl TypeFoldable for subst::Substs {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> subst::Substs {
        folder.fold_substs(self)
    }
}

impl TypeFoldable for ty::ItemSubsts {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::ItemSubsts {
        ty::ItemSubsts {
            substs: self.substs.fold_with(folder),
        }
    }
}

impl TypeFoldable for ty::AutoRef {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::AutoRef {
        folder.fold_autoref(self)
    }
}

impl TypeFoldable for typeck::vtable_origin {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> typeck::vtable_origin {
        match *self {
            typeck::vtable_static(def_id, ref substs, ref origins) => {
                let r_substs = substs.fold_with(folder);
                let r_origins = origins.fold_with(folder);
                typeck::vtable_static(def_id, r_substs, r_origins)
            }
            typeck::vtable_param(n, b) => {
                typeck::vtable_param(n, b)
            }
            typeck::vtable_error => {
                typeck::vtable_error
            }
        }
    }
}

impl TypeFoldable for ty::BuiltinBounds {
    fn fold_with<F:TypeFolder>(&self, _folder: &mut F) -> ty::BuiltinBounds {
        *self
    }
}

impl TypeFoldable for ty::ParamBounds {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::ParamBounds {
        ty::ParamBounds {
            builtin_bounds: self.builtin_bounds.fold_with(folder),
            trait_bounds: self.trait_bounds.fold_with(folder),
        }
    }
}

impl TypeFoldable for ty::TypeParameterDef {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::TypeParameterDef {
        ty::TypeParameterDef {
            ident: self.ident,
            def_id: self.def_id,
            space: self.space,
            index: self.index,
            bounds: self.bounds.fold_with(folder),
            default: self.default.fold_with(folder),
        }
    }
}

impl TypeFoldable for ty::RegionParameterDef {
    fn fold_with<F:TypeFolder>(&self, _folder: &mut F) -> ty::RegionParameterDef {
        *self
    }
}

impl TypeFoldable for ty::Generics {
    fn fold_with<F:TypeFolder>(&self, folder: &mut F) -> ty::Generics {
        ty::Generics {
            types: self.types.fold_with(folder),
            regions: self.regions.fold_with(folder),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// "super" routines: these are the default implementations for TypeFolder.
//
// They should invoke `foo.fold_with()` to do recursive folding.

pub fn super_fold_ty<T:TypeFolder>(this: &mut T,
                                   t: ty::t)
                                   -> ty::t {
    let sty = ty::get(t).sty.fold_with(this);
    ty::mk_t(this.tcx(), sty)
}

pub fn super_fold_substs<T:TypeFolder>(this: &mut T,
                                       substs: &subst::Substs)
                                       -> subst::Substs {
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

pub fn super_fold_sig<T:TypeFolder>(this: &mut T,
                                    sig: &ty::FnSig)
                                    -> ty::FnSig {
    ty::FnSig { binder_id: sig.binder_id,
                inputs: sig.inputs.fold_with(this),
                output: sig.output.fold_with(this),
                variadic: sig.variadic }
}

pub fn super_fold_bare_fn_ty<T:TypeFolder>(this: &mut T,
                                           fty: &ty::BareFnTy)
                                           -> ty::BareFnTy
{
    ty::BareFnTy { sig: fty.sig.fold_with(this),
                   abi: fty.abi,
                   fn_style: fty.fn_style }
}

pub fn super_fold_closure_ty<T:TypeFolder>(this: &mut T,
                                           fty: &ty::ClosureTy)
                                           -> ty::ClosureTy
{
    ty::ClosureTy {
        store: fty.store.fold_with(this),
        sig: fty.sig.fold_with(this),
        fn_style: fty.fn_style,
        onceness: fty.onceness,
        bounds: fty.bounds,
    }
}

pub fn super_fold_trait_ref<T:TypeFolder>(this: &mut T,
                                          t: &ty::TraitRef)
                                          -> ty::TraitRef {
    ty::TraitRef {
        def_id: t.def_id,
        substs: t.substs.fold_with(this),
    }
}

pub fn super_fold_mt<T:TypeFolder>(this: &mut T,
                                   mt: &ty::mt) -> ty::mt {
    ty::mt {ty: mt.ty.fold_with(this),
            mutbl: mt.mutbl}
}

pub fn super_fold_sty<T:TypeFolder>(this: &mut T,
                                    sty: &ty::sty) -> ty::sty {
    match *sty {
        ty::ty_box(typ) => {
            ty::ty_box(typ.fold_with(this))
        }
        ty::ty_uniq(typ) => {
            ty::ty_uniq(typ.fold_with(this))
        }
        ty::ty_ptr(ref tm) => {
            ty::ty_ptr(tm.fold_with(this))
        }
        ty::ty_vec(ref tm, sz) => {
            ty::ty_vec(tm.fold_with(this), sz)
        }
        ty::ty_enum(tid, ref substs) => {
            ty::ty_enum(tid, substs.fold_with(this))
        }
        ty::ty_trait(box ty::TyTrait {
                def_id,
                ref substs,
                bounds
            }) => {
            ty::ty_trait(box ty::TyTrait {
                def_id: def_id,
                substs: substs.fold_with(this),
                bounds: bounds
            })
        }
        ty::ty_tup(ref ts) => {
            ty::ty_tup(ts.fold_with(this))
        }
        ty::ty_bare_fn(ref f) => {
            ty::ty_bare_fn(f.fold_with(this))
        }
        ty::ty_closure(ref f) => {
            ty::ty_closure(box f.fold_with(this))
        }
        ty::ty_rptr(r, ref tm) => {
            ty::ty_rptr(r.fold_with(this), tm.fold_with(this))
        }
        ty::ty_struct(did, ref substs) => {
            ty::ty_struct(did, substs.fold_with(this))
        }
        ty::ty_nil | ty::ty_bot | ty::ty_bool | ty::ty_char | ty::ty_str |
        ty::ty_int(_) | ty::ty_uint(_) | ty::ty_float(_) |
        ty::ty_err | ty::ty_infer(_) |
        ty::ty_param(..) => {
            (*sty).clone()
        }
    }
}

pub fn super_fold_trait_store<T:TypeFolder>(this: &mut T,
                                            trait_store: ty::TraitStore)
                                            -> ty::TraitStore {
    match trait_store {
        ty::UniqTraitStore => ty::UniqTraitStore,
        ty::RegionTraitStore(r, m) => {
            ty::RegionTraitStore(r.fold_with(this), m)
        }
    }
}

pub fn super_fold_autoref<T:TypeFolder>(this: &mut T,
                                        autoref: &ty::AutoRef)
                                        -> ty::AutoRef
{
    match *autoref {
        ty::AutoPtr(r, m) => ty::AutoPtr(r.fold_with(this), m),
        ty::AutoBorrowVec(r, m) => ty::AutoBorrowVec(r.fold_with(this), m),
        ty::AutoBorrowVecRef(r, m) => ty::AutoBorrowVecRef(r.fold_with(this), m),
        ty::AutoUnsafe(m) => ty::AutoUnsafe(m),
        ty::AutoBorrowObj(r, m) => ty::AutoBorrowObj(r.fold_with(this), m),
    }
}

pub fn super_fold_item_substs<T:TypeFolder>(this: &mut T,
                                            substs: ty::ItemSubsts)
                                            -> ty::ItemSubsts
{
    ty::ItemSubsts {
        substs: substs.substs.fold_with(this),
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'a> {
    pub tcx: &'a ty::ctxt,
    pub fldop: |ty::t|: 'a -> ty::t,
}

impl<'a> TypeFolder for BottomUpFolder<'a> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt { self.tcx }

    fn fold_ty(&mut self, ty: ty::t) -> ty::t {
        let t1 = super_fold_ty(self, ty);
        (self.fldop)(t1)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

pub struct RegionFolder<'a> {
    tcx: &'a ty::ctxt,
    fld_t: |ty::t|: 'a -> ty::t,
    fld_r: |ty::Region|: 'a -> ty::Region,
}

impl<'a> RegionFolder<'a> {
    pub fn general(tcx: &'a ty::ctxt,
                   fld_r: |ty::Region|: 'a -> ty::Region,
                   fld_t: |ty::t|: 'a -> ty::t)
                   -> RegionFolder<'a> {
        RegionFolder {
            tcx: tcx,
            fld_t: fld_t,
            fld_r: fld_r
        }
    }

    pub fn regions(tcx: &'a ty::ctxt, fld_r: |ty::Region|: 'a -> ty::Region)
                   -> RegionFolder<'a> {
        fn noop(t: ty::t) -> ty::t { t }

        RegionFolder {
            tcx: tcx,
            fld_t: noop,
            fld_r: fld_r
        }
    }
}

impl<'a> TypeFolder for RegionFolder<'a> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt { self.tcx }

    fn fold_ty(&mut self, ty: ty::t) -> ty::t {
        debug!("RegionFolder.fold_ty({})", ty.repr(self.tcx()));
        let t1 = super_fold_ty(self, ty);
        (self.fld_t)(t1)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        debug!("RegionFolder.fold_region({})", r.repr(self.tcx()));
        (self.fld_r)(r)
    }
}
