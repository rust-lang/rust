// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Generalized type folding mechanism. The setup is a bit convoluted
 * but allows for convenient usage. Let T be an instance of some
 * "foldable type" (one which implements `TypeFoldable`) and F be an
 * instance of a "folder" (a type which implements `TypeFolder`). Then
 * the setup is intended to be:
 *
 *     T.fold_with(F) --calls--> F.fold_T(T) --calls--> super_fold_T(F, T)
 *
 * This way, when you define a new folder F, you can override
 * `fold_T()` to customize the behavior, and invoke `super_fold_T()`
 * to get the original behavior. Meanwhile, to actually fold
 * something, you can just write `T.fold_with(F)`, which is
 * convenient. (Note that `fold_with` will also transparently handle
 * things like a `Vec<T>` where T is foldable and so on.)
 *
 * In this ideal setup, the only function that actually *does*
 * anything is `super_fold_T`, which traverses the type `T`. Moreover,
 * `super_fold_T` should only ever call `T.fold_with()`.
 *
 * In some cases, we follow a degenerate pattern where we do not have
 * a `fold_T` nor `super_fold_T` method. Instead, `T.fold_with`
 * traverses the structure directly. This is suboptimal because the
 * behavior cannot be overriden, but it's much less work to implement.
 * If you ever *do* need an override that doesn't exist, it's not hard
 * to convert the degenerate pattern into the proper thing.
 */

use middle::subst;
use middle::subst::VecPerParamSpace;
use middle::ty;
use middle::traits;
use middle::typeck;
use std::rc::Rc;
use syntax::ast;
use syntax::owned_slice::OwnedSlice;
use util::ppaux::Repr;

///////////////////////////////////////////////////////////////////////////
// Two generic traits

/// The TypeFoldable trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in TypeFolder.
pub trait TypeFoldable {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self;
}

/// The TypeFolder trait defines the actual *folding*. There is a
/// method defined for every foldable type. Each of these has a
/// default implementation that does an "identity" fold. Within each
/// identity fold, it should invoke `foo.fold_with(self)` to fold each
/// sub-item.
pub trait TypeFolder<'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx>;

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

    fn fold_existential_bounds(&mut self, s: ty::ExistentialBounds)
                               -> ty::ExistentialBounds {
        super_fold_existential_bounds(self, s)
    }

    fn fold_autoref(&mut self, ar: &ty::AutoRef) -> ty::AutoRef {
        super_fold_autoref(self, ar)
    }

    fn fold_item_substs(&mut self, i: ty::ItemSubsts) -> ty::ItemSubsts {
        super_fold_item_substs(self, i)
    }

    fn fold_obligation(&mut self, o: &traits::Obligation) -> traits::Obligation {
        super_fold_obligation(self, o)
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

impl TypeFoldable for () {
    fn fold_with<'tcx, F:TypeFolder<'tcx>>(&self, _: &mut F) -> () {
        ()
    }
}

impl<T:TypeFoldable> TypeFoldable for Option<T> {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Option<T> {
        self.as_ref().map(|t| t.fold_with(folder))
    }
}

impl<T:TypeFoldable> TypeFoldable for Rc<T> {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Rc<T> {
        Rc::new((**self).fold_with(folder))
    }
}

impl<T:TypeFoldable> TypeFoldable for Vec<T> {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Vec<T> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<T:TypeFoldable> TypeFoldable for OwnedSlice<T> {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> OwnedSlice<T> {
        self.iter().map(|t| t.fold_with(folder)).collect()
    }
}

impl<T:TypeFoldable> TypeFoldable for VecPerParamSpace<T> {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> VecPerParamSpace<T> {
        self.map(|t| t.fold_with(folder))
    }
}

impl TypeFoldable for ty::TraitStore {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TraitStore {
        folder.fold_trait_store(*self)
    }
}

impl TypeFoldable for ty::t {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::t {
        folder.fold_ty(*self)
    }
}

impl TypeFoldable for ty::BareFnTy {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::BareFnTy {
        folder.fold_bare_fn_ty(self)
    }
}

impl TypeFoldable for ty::ClosureTy {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ClosureTy {
        folder.fold_closure_ty(self)
    }
}

impl TypeFoldable for ty::mt {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::mt {
        folder.fold_mt(self)
    }
}

impl TypeFoldable for ty::FnSig {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::FnSig {
        folder.fold_sig(self)
    }
}

impl TypeFoldable for ty::sty {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::sty {
        folder.fold_sty(self)
    }
}

impl TypeFoldable for ty::TraitRef {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TraitRef {
        folder.fold_trait_ref(self)
    }
}

impl TypeFoldable for ty::Region {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Region {
        folder.fold_region(*self)
    }
}

impl TypeFoldable for subst::Substs {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> subst::Substs {
        folder.fold_substs(self)
    }
}

impl TypeFoldable for ty::ItemSubsts {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ItemSubsts {
        ty::ItemSubsts {
            substs: self.substs.fold_with(folder),
        }
    }
}

impl TypeFoldable for ty::AutoRef {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::AutoRef {
        folder.fold_autoref(self)
    }
}

impl TypeFoldable for typeck::vtable_origin {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> typeck::vtable_origin {
        match *self {
            typeck::vtable_static(def_id, ref substs, ref origins) => {
                let r_substs = substs.fold_with(folder);
                let r_origins = origins.fold_with(folder);
                typeck::vtable_static(def_id, r_substs, r_origins)
            }
            typeck::vtable_param(n, b) => {
                typeck::vtable_param(n, b)
            }
            typeck::vtable_unboxed_closure(def_id) => {
                typeck::vtable_unboxed_closure(def_id)
            }
            typeck::vtable_error => {
                typeck::vtable_error
            }
        }
    }
}

impl TypeFoldable for ty::BuiltinBounds {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, _folder: &mut F) -> ty::BuiltinBounds {
        *self
    }
}

impl TypeFoldable for ty::ExistentialBounds {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ExistentialBounds {
        folder.fold_existential_bounds(*self)
    }
}

impl TypeFoldable for ty::ParamBounds {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::ParamBounds {
        ty::ParamBounds {
            region_bounds: self.region_bounds.fold_with(folder),
            builtin_bounds: self.builtin_bounds.fold_with(folder),
            trait_bounds: self.trait_bounds.fold_with(folder),
        }
    }
}

impl TypeFoldable for ty::TypeParameterDef {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::TypeParameterDef {
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
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::RegionParameterDef {
        ty::RegionParameterDef {
            name: self.name,
            def_id: self.def_id,
            space: self.space,
            index: self.index,
            bounds: self.bounds.fold_with(folder)
        }
    }
}

impl TypeFoldable for ty::Generics {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::Generics {
        ty::Generics {
            types: self.types.fold_with(folder),
            regions: self.regions.fold_with(folder),
        }
    }
}

impl TypeFoldable for ty::UnsizeKind {
    fn fold_with<'tcx, F: TypeFolder<'tcx>>(&self, folder: &mut F) -> ty::UnsizeKind {
        match *self {
            ty::UnsizeLength(len) => ty::UnsizeLength(len),
            ty::UnsizeStruct(box ref k, n) => ty::UnsizeStruct(box k.fold_with(folder), n),
            ty::UnsizeVtable(ty::TyTrait{bounds, def_id, substs: ref substs}, self_ty) => {
                ty::UnsizeVtable(
                    ty::TyTrait {
                        bounds: bounds.fold_with(folder),
                        def_id: def_id,
                        substs: substs.fold_with(folder)
                    },
                    self_ty.fold_with(folder))
            }
        }
    }
}

impl TypeFoldable for traits::Obligation {
    fn fold_with<'tcx, F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::Obligation {
        folder.fold_obligation(self)
    }
}

impl<N:TypeFoldable> TypeFoldable for traits::VtableImpl<N> {
    fn fold_with<'tcx, F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableImpl<N> {
        traits::VtableImpl {
            impl_def_id: self.impl_def_id,
            substs: self.substs.fold_with(folder),
            nested: self.nested.fold_with(folder),
        }
    }
}

impl<N:TypeFoldable> TypeFoldable for traits::Vtable<N> {
    fn fold_with<'tcx, F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::Vtable<N> {
        match *self {
            traits::VtableImpl(ref v) => traits::VtableImpl(v.fold_with(folder)),
            traits::VtableUnboxedClosure(d) => traits::VtableUnboxedClosure(d),
            traits::VtableParam(ref p) => traits::VtableParam(p.fold_with(folder)),
            traits::VtableBuiltin => traits::VtableBuiltin,
        }
    }
}

impl TypeFoldable for traits::VtableParam {
    fn fold_with<'tcx, F:TypeFolder<'tcx>>(&self, folder: &mut F) -> traits::VtableParam {
        traits::VtableParam {
            bound: self.bound.fold_with(folder),
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// "super" routines: these are the default implementations for TypeFolder.
//
// They should invoke `foo.fold_with()` to do recursive folding.
pub fn super_fold_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                t: ty::t)
                                                -> ty::t {
    let sty = ty::get(t).sty.fold_with(this);
    ty::mk_t(this.tcx(), sty)
}

pub fn super_fold_substs<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
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

pub fn super_fold_sig<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                 sig: &ty::FnSig)
                                                 -> ty::FnSig {
    ty::FnSig { binder_id: sig.binder_id,
                inputs: sig.inputs.fold_with(this),
                output: sig.output.fold_with(this),
                variadic: sig.variadic }
}

pub fn super_fold_bare_fn_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                        fty: &ty::BareFnTy)
                                                        -> ty::BareFnTy
{
    ty::BareFnTy { sig: fty.sig.fold_with(this),
                   abi: fty.abi,
                   fn_style: fty.fn_style }
}

pub fn super_fold_closure_ty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                        fty: &ty::ClosureTy)
                                                        -> ty::ClosureTy
{
    ty::ClosureTy {
        store: fty.store.fold_with(this),
        sig: fty.sig.fold_with(this),
        fn_style: fty.fn_style,
        onceness: fty.onceness,
        bounds: fty.bounds.fold_with(this),
        abi: fty.abi,
    }
}
pub fn super_fold_trait_ref<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                       t: &ty::TraitRef)
                                                       -> ty::TraitRef {
    ty::TraitRef {
        def_id: t.def_id,
        substs: t.substs.fold_with(this),
    }
}

pub fn super_fold_mt<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                mt: &ty::mt) -> ty::mt {
    ty::mt {ty: mt.ty.fold_with(this),
            mutbl: mt.mutbl}
}

pub fn super_fold_sty<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
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
        ty::ty_vec(typ, sz) => {
            ty::ty_vec(typ.fold_with(this), sz)
        }
        ty::ty_open(typ) => {
            ty::ty_open(typ.fold_with(this))
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
                bounds: this.fold_existential_bounds(bounds),
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
        ty::ty_unboxed_closure(did, ref region) => {
            ty::ty_unboxed_closure(did, region.fold_with(this))
        }
        ty::ty_nil | ty::ty_bot | ty::ty_bool | ty::ty_char | ty::ty_str |
        ty::ty_int(_) | ty::ty_uint(_) | ty::ty_float(_) |
        ty::ty_err | ty::ty_infer(_) |
        ty::ty_param(..) => {
            (*sty).clone()
        }
    }
}

pub fn super_fold_trait_store<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                         trait_store: ty::TraitStore)
                                                         -> ty::TraitStore {
    match trait_store {
        ty::UniqTraitStore => ty::UniqTraitStore,
        ty::RegionTraitStore(r, m) => {
            ty::RegionTraitStore(r.fold_with(this), m)
        }
    }
}

pub fn super_fold_existential_bounds<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                                bounds: ty::ExistentialBounds)
                                                                -> ty::ExistentialBounds {
    ty::ExistentialBounds {
        region_bound: bounds.region_bound.fold_with(this),
        builtin_bounds: bounds.builtin_bounds,
    }
}

pub fn super_fold_autoref<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                     autoref: &ty::AutoRef)
                                                     -> ty::AutoRef
{
    match *autoref {
        ty::AutoPtr(r, m, None) => ty::AutoPtr(this.fold_region(r), m, None),
        ty::AutoPtr(r, m, Some(ref a)) => {
            ty::AutoPtr(this.fold_region(r), m, Some(box super_fold_autoref(this, &**a)))
        }
        ty::AutoUnsafe(m, None) => ty::AutoUnsafe(m, None),
        ty::AutoUnsafe(m, Some(ref a)) => {
            ty::AutoUnsafe(m, Some(box super_fold_autoref(this, &**a)))
        }
        ty::AutoUnsize(ref k) => ty::AutoUnsize(k.fold_with(this)),
        ty::AutoUnsizeUniq(ref k) => ty::AutoUnsizeUniq(k.fold_with(this)),
    }
}

pub fn super_fold_item_substs<'tcx, T: TypeFolder<'tcx>>(this: &mut T,
                                                         substs: ty::ItemSubsts)
                                                         -> ty::ItemSubsts
{
    ty::ItemSubsts {
        substs: substs.substs.fold_with(this),
    }
}

pub fn super_fold_obligation<'tcx, T:TypeFolder<'tcx>>(this: &mut T,
                                                       obligation: &traits::Obligation)
                                                       -> traits::Obligation
{
    traits::Obligation {
        cause: obligation.cause,
        recursion_depth: obligation.recursion_depth,
        trait_ref: obligation.trait_ref.fold_with(this),
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'a, 'tcx: 'a> {
    pub tcx: &'a ty::ctxt<'tcx>,
    pub fldop: |ty::t|: 'a -> ty::t,
}

impl<'a, 'tcx> TypeFolder<'tcx> for BottomUpFolder<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.tcx }

    fn fold_ty(&mut self, ty: ty::t) -> ty::t {
        let t1 = super_fold_ty(self, ty);
        (self.fldop)(t1)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

/// Folds over the substructure of a type, visiting its component
/// types and all regions that occur *free* within it.
///
/// That is, `ty::t` can contain function or method types that bind
/// regions at the call site (`ReLateBound`), and occurrences of
/// regions (aka "lifetimes") that are bound within a type are not
/// visited by this folder; only regions that occur free will be
/// visited by `fld_r`.
///
/// (The distinction between "free" and "bound" is represented by
/// keeping track of each `FnSig` in the lexical context of the
/// current position of the fold.)
pub struct RegionFolder<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    fld_t: |ty::t|: 'a -> ty::t,
    fld_r: |ty::Region|: 'a -> ty::Region,
    within_binder_ids: Vec<ast::NodeId>,
}

impl<'a, 'tcx> RegionFolder<'a, 'tcx> {
    pub fn general(tcx: &'a ty::ctxt<'tcx>,
                   fld_r: |ty::Region|: 'a -> ty::Region,
                   fld_t: |ty::t|: 'a -> ty::t)
                   -> RegionFolder<'a, 'tcx> {
        RegionFolder {
            tcx: tcx,
            fld_t: fld_t,
            fld_r: fld_r,
            within_binder_ids: vec![],
        }
    }

    pub fn regions(tcx: &'a ty::ctxt<'tcx>, fld_r: |ty::Region|: 'a -> ty::Region)
                   -> RegionFolder<'a, 'tcx> {
        fn noop(t: ty::t) -> ty::t { t }

        RegionFolder {
            tcx: tcx,
            fld_t: noop,
            fld_r: fld_r,
            within_binder_ids: vec![],
        }
    }
}

/// If `ty` has `FnSig` (i.e. closure or fn), return its binder_id;
/// else None.
fn opt_binder_id_of_function(t: ty::t) -> Option<ast::NodeId> {
    match ty::get(t).sty {
        ty::ty_closure(ref f) => Some(f.sig.binder_id),
        ty::ty_bare_fn(ref f) => Some(f.sig.binder_id),
        _                     => None,
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionFolder<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.tcx }

    fn fold_ty(&mut self, ty: ty::t) -> ty::t {
        debug!("RegionFolder.fold_ty({})", ty.repr(self.tcx()));
        let opt_binder_id = opt_binder_id_of_function(ty);
        match opt_binder_id {
            Some(binder_id) => self.within_binder_ids.push(binder_id),
            None => {}
        }

        let t1 = super_fold_ty(self, ty);
        let ret = (self.fld_t)(t1);

        if opt_binder_id.is_some() {
            self.within_binder_ids.pop();
        }

        ret
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReLateBound(binder_id, _) if self.within_binder_ids.contains(&binder_id) => {
                debug!("RegionFolder.fold_region({}) skipped bound region", r.repr(self.tcx()));
                r
            }
            _ => {
                debug!("RegionFolder.fold_region({}) folding free region", r.repr(self.tcx()));
                (self.fld_r)(r)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Region eraser
//
// Replaces all free regions with 'static. Useful in trans.

pub struct RegionEraser<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

pub fn erase_regions<T:TypeFoldable>(tcx: &ty::ctxt, t: T) -> T {
    let mut eraser = RegionEraser { tcx: tcx };
    t.fold_with(&mut eraser)
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionEraser<'a, 'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> { self.tcx }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        match r {
            ty::ReLateBound(..) | ty::ReEarlyBound(..) => r,
            _ => ty::ReStatic
        }
    }
}
