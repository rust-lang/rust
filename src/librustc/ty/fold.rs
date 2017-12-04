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

use middle::const_val::ConstVal;
use hir::def_id::DefId;
use ty::{self, Binder, Ty, TyCtxt, TypeFlags};

use std::fmt;
use std::collections::BTreeMap;
use util::nodemap::FxHashSet;

/// The TypeFoldable trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in TypeFolder.
pub trait TypeFoldable<'tcx>: fmt::Debug + Clone {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self;
    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
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
    fn has_projections(&self) -> bool {
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
    fn has_re_skol(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_SKOL)
    }
    fn has_closure_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_CLOSURE)
    }
    /// "Free" regions in this context means that it has any region
    /// that is not (a) erased or (b) late-bound.
    fn has_free_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    /// True if there any any un-erased free regions.
    fn has_erasable_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    fn is_normalized_for_trans(&self) -> bool {
        !self.has_type_flags(TypeFlags::HAS_RE_INFER |
                             TypeFlags::HAS_FREE_REGIONS |
                             TypeFlags::HAS_TY_INFER |
                             TypeFlags::HAS_PARAMS |
                             TypeFlags::HAS_NORMALIZABLE_PROJECTION |
                             TypeFlags::HAS_TY_ERR |
                             TypeFlags::HAS_SELF)
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
pub trait TypeFolder<'gcx: 'tcx, 'tcx> : Sized {
    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx>;

    fn fold_binder<T>(&mut self, t: &Binder<T>) -> Binder<T>
        where T : TypeFoldable<'tcx>
    {
        t.super_fold_with(self)
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        t.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        r.super_fold_with(self)
    }

    fn fold_const(&mut self, c: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        c.super_fold_with(self)
    }
}

pub trait TypeVisitor<'tcx> : Sized {
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> bool {
        t.super_visit_with(self)
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        r.super_visit_with(self)
    }

    fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> bool {
        c.super_visit_with(self)
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a, F>
    where F: FnMut(Ty<'tcx>) -> Ty<'tcx>
{
    pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
    pub fldop: F,
}

impl<'a, 'gcx, 'tcx, F> TypeFolder<'gcx, 'tcx> for BottomUpFolder<'a, 'gcx, 'tcx, F>
    where F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
{
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t1 = ty.super_fold_with(self);
        (self.fldop)(t1)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Collects the free and escaping regions in `value` into `region_set`. Returns
    /// whether any late-bound regions were skipped
    pub fn collect_regions<T>(self,
        value: &T,
        region_set: &mut FxHashSet<ty::Region<'tcx>>)
        -> bool
        where T : TypeFoldable<'tcx>
    {
        let mut have_bound_regions = false;
        self.fold_regions(value, &mut have_bound_regions, |r, d| {
            region_set.insert(self.mk_region(r.from_depth(d)));
            r
        });
        have_bound_regions
    }

    /// Folds the escaping and free regions in `value` using `f`, and
    /// sets `skipped_regions` to true if any late-bound region was found
    /// and skipped.
    pub fn fold_regions<T,F>(self,
        value: &T,
        skipped_regions: &mut bool,
        mut f: F)
        -> T
        where F : FnMut(ty::Region<'tcx>, u32) -> ty::Region<'tcx>,
              T : TypeFoldable<'tcx>,
    {
        value.fold_with(&mut RegionFolder::new(self, skipped_regions, &mut f))
    }

    pub fn for_each_free_region<T,F>(self,
                                     value: &T,
                                     callback: F)
        where F: FnMut(ty::Region<'tcx>),
              T: TypeFoldable<'tcx>,
    {
        value.visit_with(&mut RegionVisitor { current_depth: 0, callback });

        struct RegionVisitor<F> {
            current_depth: u32,
            callback: F,
        }

        impl<'tcx, F> TypeVisitor<'tcx> for RegionVisitor<F>
            where F : FnMut(ty::Region<'tcx>)
        {
            fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> bool {
                self.current_depth += 1;
                t.skip_binder().visit_with(self);
                self.current_depth -= 1;

                false // keep visiting
            }

            fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
                match *r {
                    ty::ReLateBound(debruijn, _) if debruijn.depth <= self.current_depth => {
                        /* ignore bound regions */
                    }
                    _ => (self.callback)(r),
                }

                false // keep visiting
            }
        }
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

pub struct RegionFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    skipped_regions: &'a mut bool,
    current_depth: u32,
    fld_r: &'a mut (FnMut(ty::Region<'tcx>, u32) -> ty::Region<'tcx> + 'a),
}

impl<'a, 'gcx, 'tcx> RegionFolder<'a, 'gcx, 'tcx> {
    pub fn new<F>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                  skipped_regions: &'a mut bool,
                  fld_r: &'a mut F) -> RegionFolder<'a, 'gcx, 'tcx>
        where F : FnMut(ty::Region<'tcx>, u32) -> ty::Region<'tcx>
    {
        RegionFolder {
            tcx,
            skipped_regions,
            current_depth: 1,
            fld_r,
        }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionFolder<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T> {
        self.current_depth += 1;
        let t = t.super_fold_with(self);
        self.current_depth -= 1;
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
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

struct RegionReplacer<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    current_depth: u32,
    fld_r: &'a mut (FnMut(ty::BoundRegion) -> ty::Region<'tcx> + 'a),
    map: BTreeMap<ty::BoundRegion, ty::Region<'tcx>>
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Replace all regions bound by the given `Binder` with the
    /// results returned by the closure; the closure is expected to
    /// return a free region (relative to this binder), and hence the
    /// binder is removed in the return type. The closure is invoked
    /// once for each unique `BoundRegion`; multiple references to the
    /// same `BoundRegion` will reuse the previous result.  A map is
    /// returned at the end with each bound region and the free region
    /// that replaced it.
    pub fn replace_late_bound_regions<T,F>(self,
        value: &Binder<T>,
        mut f: F)
        -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
        where F : FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
              T : TypeFoldable<'tcx>,
    {
        let mut replacer = RegionReplacer::new(self, &mut f);
        let result = value.skip_binder().fold_with(&mut replacer);
        (result, replacer.map)
    }

    /// Replace any late-bound regions bound in `value` with
    /// free variants attached to `all_outlive_scope`.
    pub fn liberate_late_bound_regions<T>(
        &self,
        all_outlive_scope: DefId,
        value: &ty::Binder<T>
    ) -> T
    where T: TypeFoldable<'tcx> {
        self.replace_late_bound_regions(value, |br| {
            self.mk_region(ty::ReFree(ty::FreeRegion {
                scope: all_outlive_scope,
                bound_region: br
            }))
        }).0
    }

    /// Flattens two binding levels into one. So `for<'a> for<'b> Foo`
    /// becomes `for<'a,'b> Foo`.
    pub fn flatten_late_bound_regions<T>(self, bound2_value: &Binder<Binder<T>>)
                                         -> Binder<T>
        where T: TypeFoldable<'tcx>
    {
        let bound0_value = bound2_value.skip_binder().skip_binder();
        let value = self.fold_regions(bound0_value, &mut false,
                                      |region, current_depth| {
            match *region {
                ty::ReLateBound(debruijn, br) if debruijn.depth >= current_depth => {
                    // should be true if no escaping regions from bound2_value
                    assert!(debruijn.depth - current_depth <= 1);
                    self.mk_region(ty::ReLateBound(ty::DebruijnIndex::new(current_depth), br))
                }
                _ => {
                    region
                }
            }
        });
        Binder(value)
    }

    /// Returns a set of all late-bound regions that are constrained
    /// by `value`, meaning that if we instantiate those LBR with
    /// variables and equate `value` with something else, those
    /// variables will also be equated.
    pub fn collect_constrained_late_bound_regions<T>(&self, value: &Binder<T>)
                                                     -> FxHashSet<ty::BoundRegion>
        where T : TypeFoldable<'tcx>
    {
        self.collect_late_bound_regions(value, true)
    }

    /// Returns a set of all late-bound regions that appear in `value` anywhere.
    pub fn collect_referenced_late_bound_regions<T>(&self, value: &Binder<T>)
                                                    -> FxHashSet<ty::BoundRegion>
        where T : TypeFoldable<'tcx>
    {
        self.collect_late_bound_regions(value, false)
    }

    fn collect_late_bound_regions<T>(&self, value: &Binder<T>, just_constraint: bool)
                                     -> FxHashSet<ty::BoundRegion>
        where T : TypeFoldable<'tcx>
    {
        let mut collector = LateBoundRegionsCollector::new(just_constraint);
        let result = value.skip_binder().visit_with(&mut collector);
        assert!(!result); // should never have stopped early
        collector.regions
    }

    /// Replace any late-bound regions bound in `value` with `'erased`. Useful in trans but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn erase_late_bound_regions<T>(self, value: &Binder<T>) -> T
        where T : TypeFoldable<'tcx>
    {
        self.replace_late_bound_regions(value, |_| self.types.re_erased).0
    }

    /// Rewrite any late-bound regions so that they are anonymous.  Region numbers are
    /// assigned starting at 1 and increasing monotonically in the order traversed
    /// by the fold operation.
    ///
    /// The chief purpose of this function is to canonicalize regions so that two
    /// `FnSig`s or `TraitRef`s which are equivalent up to region naming will become
    /// structurally identical.  For example, `for<'a, 'b> fn(&'a isize, &'b isize)` and
    /// `for<'a, 'b> fn(&'b isize, &'a isize)` will become identical after anonymization.
    pub fn anonymize_late_bound_regions<T>(self, sig: &Binder<T>) -> Binder<T>
        where T : TypeFoldable<'tcx>,
    {
        let mut counter = 0;
        Binder(self.replace_late_bound_regions(sig, |_| {
            counter += 1;
            self.mk_region(ty::ReLateBound(ty::DebruijnIndex::new(1), ty::BrAnon(counter)))
        }).0)
    }
}

impl<'a, 'gcx, 'tcx> RegionReplacer<'a, 'gcx, 'tcx> {
    fn new<F>(tcx: TyCtxt<'a, 'gcx, 'tcx>, fld_r: &'a mut F)
              -> RegionReplacer<'a, 'gcx, 'tcx>
        where F : FnMut(ty::BoundRegion) -> ty::Region<'tcx>
    {
        RegionReplacer {
            tcx,
            current_depth: 1,
            fld_r,
            map: BTreeMap::default()
        }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionReplacer<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>) -> ty::Binder<T> {
        self.current_depth += 1;
        let t = t.super_fold_with(self);
        self.current_depth -= 1;
        t
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_regions_escaping_depth(self.current_depth-1) {
            return t;
        }

        t.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, br) if debruijn.depth == self.current_depth => {
                let fld_r = &mut self.fld_r;
                let region = *self.map.entry(br).or_insert_with(|| fld_r(br));
                if let ty::ReLateBound(debruijn1, br) = *region {
                    // If the callback returns a late-bound region,
                    // that region should always use depth 1. Then we
                    // adjust it to the correct depth.
                    assert_eq!(debruijn1.depth, 1);
                    self.tcx.mk_region(ty::ReLateBound(debruijn, br))
                } else {
                    region
                }
            }
            _ => r
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

pub fn shift_region(region: ty::RegionKind, amount: u32) -> ty::RegionKind {
    match region {
        ty::ReLateBound(debruijn, br) => {
            ty::ReLateBound(debruijn.shifted(amount), br)
        }
        _ => {
            region
        }
    }
}

pub fn shift_region_ref<'a, 'gcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    region: ty::Region<'tcx>,
    amount: u32)
    -> ty::Region<'tcx>
{
    match region {
        &ty::ReLateBound(debruijn, br) if amount > 0 => {
            tcx.mk_region(ty::ReLateBound(debruijn.shifted(amount), br))
        }
        _ => {
            region
        }
    }
}

pub fn shift_regions<'a, 'gcx, 'tcx, T>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                        amount: u32,
                                        value: &T) -> T
    where T: TypeFoldable<'tcx>
{
    debug!("shift_regions(value={:?}, amount={})",
           value, amount);

    value.fold_with(&mut RegionFolder::new(tcx, &mut false, &mut |region, _current_depth| {
        shift_region_ref(tcx, region, amount)
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
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> bool {
        self.depth += 1;
        let result = t.super_visit_with(self);
        self.depth -= 1;
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
        t.region_depth > self.depth
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        r.escapes_depth(self.depth)
    }
}

struct HasTypeFlagsVisitor {
    flags: ty::TypeFlags,
}

impl<'tcx> TypeVisitor<'tcx> for HasTypeFlagsVisitor {
    fn visit_ty(&mut self, t: Ty) -> bool {
        debug!("HasTypeFlagsVisitor: t={:?} t.flags={:?} self.flags={:?}", t, t.flags, self.flags);
        t.flags.intersects(self.flags)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        let flags = r.type_flags();
        debug!("HasTypeFlagsVisitor: r={:?} r.flags={:?} self.flags={:?}", r, flags, self.flags);
        flags.intersects(self.flags)
    }

    fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> bool {
        if let ConstVal::Unevaluated(..) = c.val {
            let projection_flags = TypeFlags::HAS_NORMALIZABLE_PROJECTION |
                TypeFlags::HAS_PROJECTION;
            if projection_flags.intersects(self.flags) {
                return true;
            }
        }
        c.super_visit_with(self)
    }
}

/// Collects all the late-bound regions it finds into a hash set.
struct LateBoundRegionsCollector {
    current_depth: u32,
    regions: FxHashSet<ty::BoundRegion>,
    just_constrained: bool,
}

impl LateBoundRegionsCollector {
    fn new(just_constrained: bool) -> Self {
        LateBoundRegionsCollector {
            current_depth: 1,
            regions: FxHashSet(),
            just_constrained,
        }
    }
}

impl<'tcx> TypeVisitor<'tcx> for LateBoundRegionsCollector {
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> bool {
        self.current_depth += 1;
        let result = t.super_visit_with(self);
        self.current_depth -= 1;
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> bool {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs to a projection, as they may not appear
        // in the normalized form
        if self.just_constrained {
            match t.sty {
                ty::TyProjection(..) | ty::TyAnon(..) => { return false; }
                _ => { }
            }
        }

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        match *r {
            ty::ReLateBound(debruijn, br) if debruijn.depth == self.current_depth => {
                self.regions.insert(br);
            }
            _ => { }
        }
        false
    }
}
