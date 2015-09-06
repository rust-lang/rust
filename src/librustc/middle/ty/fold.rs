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

use middle::region;
use middle::subst;
use middle::ty::{self, Binder, Ty, HasTypeFlags, RegionEscape};

use std::fmt;
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

    fn fold_binder<T>(&mut self, t: &Binder<T>) -> Binder<T>
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
// "super" routines: these are the default implementations for TypeFolder.
//
// They should invoke `foo.fold_with()` to do recursive folding.

pub fn super_fold_binder<'tcx, T, U>(this: &mut T,
                                     binder: &Binder<U>)
                                     -> Binder<U>
    where T : TypeFolder<'tcx>, U : TypeFoldable<'tcx>
{
    this.enter_region_binder();
    let result = Binder(binder.0.fold_with(this));
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

impl<'tcx> ty::ctxt<'tcx> {
    /// Collects the free and escaping regions in `value` into `region_set`. Returns
    /// whether any late-bound regions were skipped
    pub fn collect_regions<T>(&self,
        value: &T,
        region_set: &mut FnvHashSet<ty::Region>)
        -> bool
        where T : TypeFoldable<'tcx>
    {
        let mut have_bound_regions = false;
        self.fold_regions(value, &mut have_bound_regions,
                          |r, d| { region_set.insert(r.from_depth(d)); r });
        have_bound_regions
    }

    /// Folds the escaping and free regions in `value` using `f`, and
    /// sets `skipped_regions` to true if any late-bound region was found
    /// and skipped.
    pub fn fold_regions<T,F>(&self,
        value: &T,
        skipped_regions: &mut bool,
        mut f: F)
        -> T
        where F : FnMut(ty::Region, u32) -> ty::Region,
              T : TypeFoldable<'tcx>,
    {
        value.fold_with(&mut RegionFolder::new(self, skipped_regions, &mut f))
    }
}

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

impl<'tcx> ty::ctxt<'tcx> {
    pub fn replace_late_bound_regions<T,F>(&self,
        value: &Binder<T>,
        mut f: F)
        -> (T, FnvHashMap<ty::BoundRegion, ty::Region>)
        where F : FnMut(ty::BoundRegion) -> ty::Region,
              T : TypeFoldable<'tcx>,
    {
        debug!("replace_late_bound_regions({:?})", value);
        let mut replacer = RegionReplacer::new(self, &mut f);
        let result = value.skip_binder().fold_with(&mut replacer);
        (result, replacer.map)
    }


    /// Replace any late-bound regions bound in `value` with free variants attached to scope-id
    /// `scope_id`.
    pub fn liberate_late_bound_regions<T>(&self,
        all_outlive_scope: region::CodeExtent,
        value: &Binder<T>)
        -> T
        where T : TypeFoldable<'tcx>
    {
        self.replace_late_bound_regions(value, |br| {
            ty::ReFree(ty::FreeRegion{scope: all_outlive_scope, bound_region: br})
        }).0
    }

    /// Flattens two binding levels into one. So `for<'a> for<'b> Foo`
    /// becomes `for<'a,'b> Foo`.
    pub fn flatten_late_bound_regions<T>(&self, bound2_value: &Binder<Binder<T>>)
                                         -> Binder<T>
        where T: TypeFoldable<'tcx>
    {
        let bound0_value = bound2_value.skip_binder().skip_binder();
        let value = self.fold_regions(bound0_value, &mut false,
                                      |region, current_depth| {
            match region {
                ty::ReLateBound(debruijn, br) if debruijn.depth >= current_depth => {
                    // should be true if no escaping regions from bound2_value
                    assert!(debruijn.depth - current_depth <= 1);
                    ty::ReLateBound(ty::DebruijnIndex::new(current_depth), br)
                }
                _ => {
                    region
                }
            }
        });
        Binder(value)
    }

    pub fn no_late_bound_regions<T>(&self, value: &Binder<T>) -> Option<T>
        where T : TypeFoldable<'tcx> + RegionEscape
    {
        if value.0.has_escaping_regions() {
            None
        } else {
            Some(value.0.clone())
        }
    }

    /// Replace any late-bound regions bound in `value` with `'static`. Useful in trans but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn erase_late_bound_regions<T>(&self, value: &Binder<T>) -> T
        where T : TypeFoldable<'tcx>
    {
        self.replace_late_bound_regions(value, |_| ty::ReStatic).0
    }

    /// Rewrite any late-bound regions so that they are anonymous.  Region numbers are
    /// assigned starting at 1 and increasing monotonically in the order traversed
    /// by the fold operation.
    ///
    /// The chief purpose of this function is to canonicalize regions so that two
    /// `FnSig`s or `TraitRef`s which are equivalent up to region naming will become
    /// structurally identical.  For example, `for<'a, 'b> fn(&'a isize, &'b isize)` and
    /// `for<'a, 'b> fn(&'b isize, &'a isize)` will become identical after anonymization.
    pub fn anonymize_late_bound_regions<T>(&self, sig: &Binder<T>) -> Binder<T>
        where T : TypeFoldable<'tcx>,
    {
        let mut counter = 0;
        Binder(self.replace_late_bound_regions(sig, |_| {
            counter += 1;
            ty::ReLateBound(ty::DebruijnIndex::new(1), ty::BrAnon(counter))
        }).0)
    }
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
