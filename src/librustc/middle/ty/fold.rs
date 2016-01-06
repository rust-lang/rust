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
//!   T.fold_with(F) --calls--> F.fold_T(T) --calls--> T.super_fold_with(F)
//!
//! This way, when you define a new folder F, you can override
//! `fold_T()` to customize the behavior, and invoke `T.super_fold_with()`
//! to get the original behavior. Meanwhile, to actually fold
//! something, you can just write `T.fold_with(F)`, which is
//! convenient. (Note that `fold_with` will also transparently handle
//! things like a `Vec<T>` where T is foldable and so on.)
//!
//! In this ideal setup, the only function that actually *does*
//! anything is `T.super_fold_with()`, which traverses the type `T`.
//! Moreover, `T.super_fold_with()` should only ever call `T.fold_with()`.
//!
//! In some cases, we follow a degenerate pattern where we do not have
//! a `fold_T` method. Instead, `T.fold_with` traverses the structure directly.
//! This is suboptimal because the behavior cannot be overridden, but it's
//! much less work to implement. If you ever *do* need an override that
//! doesn't exist, it's not hard to convert the degenerate pattern into the
//! proper thing.
//!
//! A `TypeFoldable` T can also be visited by a `TypeVisitor` V using similar setup:
//!   T.visit_with(V) --calls--> V.visit_T(T) --calls--> T.super_visit_with(V).
//! These methods return true to indicate that the visitor has found what it is looking for
//! and does not need to visit anything else.

use middle::region;
use middle::subst;
use middle::ty::adjustment;
use middle::ty::{self, Binder, Ty, TypeFlags};

use std::fmt;
use util::nodemap::{FnvHashMap, FnvHashSet};

/// The TypeFoldable trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in TypeFolder.
pub trait TypeFoldable<'tcx>: fmt::Debug + Clone {
    fn super_fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self;
    fn fold_with<F: TypeFolder<'tcx>>(&self, folder: &mut F) -> Self {
        self.super_fold_with(folder)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool;
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.super_visit_with(visitor)
    }

    fn has_regions_escaping_depth(&self, depth: u32) -> bool {
        self.visit_with(&mut HasEscapingRegionsVisitor { depth: depth })
    }
    fn has_escaping_regions(&self) -> bool {
        self.has_regions_escaping_depth(0)
    }

    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.visit_with(&mut HasTypeFlagsVisitor { flags: flags })
    }
    fn has_projection_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PROJECTION)
    }
    fn references_error(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_ERR)
    }
    fn has_param_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PARAMS)
    }
    fn has_self_ty(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_SELF)
    }
    fn has_infer_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER)
    }
    fn needs_infer(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER | TypeFlags::HAS_RE_INFER)
    }
    fn needs_subst(&self) -> bool {
        self.has_type_flags(TypeFlags::NEEDS_SUBST)
    }
    fn has_closure_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_CLOSURE)
    }
    fn has_erasable_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_EARLY_BOUND |
                            TypeFlags::HAS_RE_INFER |
                            TypeFlags::HAS_FREE_REGIONS)
    }
    /// Indicates whether this value references only 'global'
    /// types/lifetimes that are the same regardless of what fn we are
    /// in. This is used for caching. Errs on the side of returning
    /// false.
    fn is_global(&self) -> bool {
        !self.has_type_flags(TypeFlags::HAS_LOCAL_NAMES)
    }
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
        t.super_fold_with(self)
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        t.super_fold_with(self)
    }

    fn fold_mt(&mut self, t: &ty::TypeAndMut<'tcx>) -> ty::TypeAndMut<'tcx> {
        t.super_fold_with(self)
    }

    fn fold_trait_ref(&mut self, t: &ty::TraitRef<'tcx>) -> ty::TraitRef<'tcx> {
        t.super_fold_with(self)
    }

    fn fold_substs(&mut self,
                   substs: &subst::Substs<'tcx>)
                   -> subst::Substs<'tcx> {
        substs.super_fold_with(self)
    }

    fn fold_fn_sig(&mut self,
                   sig: &ty::FnSig<'tcx>)
                   -> ty::FnSig<'tcx> {
        sig.super_fold_with(self)
    }

    fn fold_output(&mut self,
                      output: &ty::FnOutput<'tcx>)
                      -> ty::FnOutput<'tcx> {
        output.super_fold_with(self)
    }

    fn fold_bare_fn_ty(&mut self,
                       fty: &ty::BareFnTy<'tcx>)
                       -> ty::BareFnTy<'tcx>
    {
        fty.super_fold_with(self)
    }

    fn fold_closure_ty(&mut self,
                       fty: &ty::ClosureTy<'tcx>)
                       -> ty::ClosureTy<'tcx> {
        fty.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region) -> ty::Region {
        r.super_fold_with(self)
    }

    fn fold_existential_bounds(&mut self, s: &ty::ExistentialBounds<'tcx>)
                               -> ty::ExistentialBounds<'tcx> {
        s.super_fold_with(self)
    }

    fn fold_autoref(&mut self, ar: &adjustment::AutoRef<'tcx>)
                    -> adjustment::AutoRef<'tcx> {
        ar.super_fold_with(self)
    }
}

pub trait TypeVisitor<'tcx> : Sized {
    fn enter_region_binder(&mut self) { }
    fn exit_region_binder(&mut self) { }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region) -> bool {
        r.super_visit_with(self)
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
        let t1 = ty.super_fold_with(self);
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
        where T : TypeFoldable<'tcx>
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

        t.super_fold_with(self)
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

impl<'tcx> ty::ctxt<'tcx> {
    /// Returns an equivalent value with all free regions removed (note
    /// that late-bound regions remain, because they are important for
    /// subtyping, but they are anonymized and normalized as well)..
    pub fn erase_regions<T>(&self, value: &T) -> T
        where T : TypeFoldable<'tcx>
    {
        let value1 = value.fold_with(&mut RegionEraser(self));
        debug!("erase_regions({:?}) = {:?}",
               value, value1);
        return value1;

        struct RegionEraser<'a, 'tcx: 'a>(&'a ty::ctxt<'tcx>);

        impl<'a, 'tcx> TypeFolder<'tcx> for RegionEraser<'a, 'tcx> {
            fn tcx(&self) -> &ty::ctxt<'tcx> { self.0 }

            fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
                match self.tcx().normalized_cache.borrow().get(&ty).cloned() {
                    None => {}
                    Some(u) => return u
                }

                let t_norm = ty.super_fold_with(self);
                self.tcx().normalized_cache.borrow_mut().insert(ty, t_norm);
                return t_norm;
            }

            fn fold_binder<T>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T>
                where T : TypeFoldable<'tcx>
            {
                let u = self.tcx().anonymize_late_bound_regions(t);
                u.super_fold_with(self)
            }

            fn fold_region(&mut self, r: ty::Region) -> ty::Region {
                // because late-bound regions affect subtyping, we can't
                // erase the bound/free distinction, but we can replace
                // all free regions with 'static.
                //
                // Note that we *CAN* replace early-bound regions -- the
                // type system never "sees" those, they get substituted
                // away. In trans, they will always be erased to 'static
                // whenever a substitution occurs.
                match r {
                    ty::ReLateBound(..) => r,
                    _ => ty::ReStatic
                }
            }

            fn fold_substs(&mut self,
                           substs: &subst::Substs<'tcx>)
                           -> subst::Substs<'tcx> {
                subst::Substs { regions: subst::ErasedRegions,
                                types: substs.types.fold_with(self) }
            }
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

/// An "escaping region" is a bound region whose binder is not part of `t`.
///
/// So, for example, consider a type like the following, which has two binders:
///
///    for<'a> fn(x: for<'b> fn(&'a isize, &'b isize))
///    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ outer scope
///                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~  inner scope
///
/// This type has *bound regions* (`'a`, `'b`), but it does not have escaping regions, because the
/// binders of both `'a` and `'b` are part of the type itself. However, if we consider the *inner
/// fn type*, that type has an escaping region: `'a`.
///
/// Note that what I'm calling an "escaping region" is often just called a "free region". However,
/// we already use the term "free region". It refers to the regions that we use to represent bound
/// regions on a fn definition while we are typechecking its body.
///
/// To clarify, conceptually there is no particular difference between an "escaping" region and a
/// "free" region. However, there is a big difference in practice. Basically, when "entering" a
/// binding level, one is generally required to do some sort of processing to a bound region, such
/// as replacing it with a fresh/skolemized region, or making an entry in the environment to
/// represent the scope to which it is attached, etc. An escaping region represents a bound region
/// for which this processing has not yet been done.
struct HasEscapingRegionsVisitor {
    depth: u32,
}

impl<'tcx> TypeVisitor<'tcx> for HasEscapingRegionsVisitor {
    fn enter_region_binder(&mut self) {
        self.depth += 1;
    }

    fn exit_region_binder(&mut self) {
        self.depth -= 1;
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
        t.region_depth > self.depth
    }

    fn visit_region(&mut self, r: ty::Region) -> bool {
        r.escapes_depth(self.depth)
    }
}

struct HasTypeFlagsVisitor {
    flags: ty::TypeFlags,
}

impl<'tcx> TypeVisitor<'tcx> for HasTypeFlagsVisitor {
    fn visit_ty(&mut self, t: Ty) -> bool {
        t.flags.get().intersects(self.flags)
    }

    fn visit_region(&mut self, r: ty::Region) -> bool {
        if self.flags.intersects(ty::TypeFlags::HAS_LOCAL_NAMES) {
            // does this represent a region that cannot be named
            // in a global way? used in fulfillment caching.
            match r {
                ty::ReStatic | ty::ReEmpty => {}
                _ => return true,
            }
        }
        if self.flags.intersects(ty::TypeFlags::HAS_RE_INFER) {
            match r {
                ty::ReVar(_) | ty::ReSkolemized(..) => { return true }
                _ => {}
            }
        }
        false
    }
}
