//! Generalized type folding mechanism. The setup is a bit convoluted
//! but allows for convenient usage. Let T be an instance of some
//! "foldable type" (one which implements `TypeFoldable`) and F be an
//! instance of a "folder" (a type which implements `TypeFolder`). Then
//! the setup is intended to be:
//!
//!     T.fold_with(F) --calls--> F.fold_T(T) --calls--> T.super_fold_with(F)
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
//!
//!     T.visit_with(V) --calls--> V.visit_T(T) --calls--> T.super_visit_with(V).
//!
//! These methods return true to indicate that the visitor has found what it is
//! looking for, and does not need to visit anything else.

use crate::hir::def_id::DefId;
use crate::ty::{self, Binder, Ty, TyCtxt, TypeFlags};
use crate::ty::subst::SubstsRef;

use std::collections::BTreeMap;
use std::fmt;
use crate::util::nodemap::FxHashSet;

/// The TypeFoldable trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in TypeFolder.
///
/// To implement this conveniently, use the
/// `BraceStructTypeFoldableImpl` etc macros found in `macros.rs`.
pub trait TypeFoldable<'tcx>: fmt::Debug + Clone {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>
        (&self, folder: &mut F) -> Result<Self, F::Error>;
    fn fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>
        (&self, folder: &mut F) -> Result<Self, F::Error>
    {
        self.super_fold_with(folder)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> Result<(), V::Error>;
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> Result<(), V::Error> {
        self.super_visit_with(visitor)
    }

    /// Returns `true` if `self` has any late-bound regions that are either
    /// bound by `binder` or bound by some binder outside of `binder`.
    /// If `binder` is `ty::INNERMOST`, this indicates whether
    /// there are any late-bound regions that appear free.
    fn has_vars_bound_at_or_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.visit_with(&mut HasEscapingVarsVisitor { outer_index: binder }).is_err()
    }

    /// Returns `true` if this `self` has any regions that escape `binder` (and
    /// hence are not bound by it).
    fn has_vars_bound_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.has_vars_bound_at_or_above(binder.shifted_in(1))
    }

    fn has_escaping_bound_vars(&self) -> bool {
        self.has_vars_bound_at_or_above(ty::INNERMOST)
    }

    fn has_type_flags(&self, flags: TypeFlags) -> bool {
        self.visit_with(&mut HasTypeFlagsVisitor { flags }).is_err()
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
    fn has_placeholders(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_PLACEHOLDER | TypeFlags::HAS_TY_PLACEHOLDER)
    }
    fn needs_subst(&self) -> bool {
        self.has_type_flags(TypeFlags::NEEDS_SUBST)
    }
    fn has_re_placeholders(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_PLACEHOLDER)
    }
    fn has_closure_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_CLOSURE)
    }
    /// "Free" regions in this context means that it has any region
    /// that is not (a) erased or (b) late-bound.
    fn has_free_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    /// True if there are any un-erased free regions.
    fn has_erasable_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    /// Indicates whether this value references only 'global'
    /// types/lifetimes that are the same regardless of what fn we are
    /// in. This is used for caching.
    fn is_global(&self) -> bool {
        !self.has_type_flags(TypeFlags::HAS_FREE_LOCAL_NAMES)
    }

    /// True if there are any late-bound regions
    fn has_late_bound_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_LATE_BOUND)
    }

    /// A visitor that does not recurse into types, works like `fn walk_shallow` in `Ty`.
    fn visit_tys_shallow(&self, visit: impl FnMut(Ty<'tcx>) -> bool) -> bool {

        pub struct Visitor<F>(F);

        impl<'tcx, F: FnMut(Ty<'tcx>) -> bool> TypeVisitor<'tcx> for Visitor<F> {
            type Error = ();

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Result<(), ()> {
                match self.0(ty) {
                    true => Err(()),
                    false => Ok(())
                }
            }
        }

        self.visit_with(&mut Visitor(visit)).is_err()
    }
}

/// The `TypeFolder` trait defines the actual *folding*. There is a
/// method defined for every foldable type. Each of these has a
/// default implementation that does an "identity" fold. Within each
/// identity fold, it should invoke `foo.fold_with(self)` to fold each
/// sub-item.
pub trait TypeFolder<'gcx: 'tcx, 'tcx> : Sized {
    type Error;

    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx>;

    fn fold_binder<T>(&mut self, t: &Binder<T>) -> Result<Binder<T>, Self::Error>
        where T : TypeFoldable<'tcx>
    {
        t.super_fold_with(self)
    }

    #[inline]
    /// If `false` - the default - then `ty::Invariant` might be used instead of the
    /// correct variance when folding an item with a variance.
    ///
    /// Otherwise, the correct variance is looked up from the tcx, which can
    /// be a performance and cycle hazard.
    fn use_variances(&self) -> bool {
        false
    }

    #[inline]
    fn fold_item_substs(&mut self, item_def_id: DefId, substs: SubstsRef<'tcx>)
                        -> Result<SubstsRef<'tcx>, Self::Error>
    {
        if self.use_variances() {
            let variances = self.tcx().variances_of(item_def_id);
            ty::subst::fold_with_variances(self, &variances, substs)
        } else {
            substs.fold_with(self)
        }
    }

    #[inline]
    fn fold_with_variance<T>(&mut self, _variance: ty::Variance, t: &T)
                             -> Result<T, Self::Error>
        where T : TypeFoldable<'tcx>
    {
        t.fold_with(self)
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        t.super_fold_with(self)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, Self::Error> {
        r.super_fold_with(self)
    }

    fn fold_const(&mut self, c: &'tcx ty::LazyConst<'tcx>)
                  -> Result<&'tcx ty::LazyConst<'tcx>, Self::Error>
    {
        c.super_fold_with(self)
    }
}

pub trait TypeVisitor<'tcx> : Sized {
    type Error;

    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> Result<(), Self::Error> {
        t.super_visit_with(self)
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Result<(), Self::Error> {
        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> Result<(), Self::Error> {
        r.super_visit_with(self)
    }

    fn visit_const(&mut self, c: &'tcx ty::LazyConst<'tcx>) -> Result<(), Self::Error> {
        c.super_visit_with(self)
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a, F, G>
    where F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
          G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
{
    pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
    pub fldop: F,
    pub reg_op: G,
}

impl<'a, 'gcx, 'tcx, F, G> TypeFolder<'gcx, 'tcx> for BottomUpFolder<'a, 'gcx, 'tcx, F, G>
    where F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
          G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
{
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        let t1 = ty.super_fold_with(self)?;
        Ok((self.fldop)(t1))
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        let r = r.super_fold_with(self)?;
        Ok((self.reg_op)(r))
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
            region_set.insert(self.mk_region(r.shifted_out_to_binder(d)));
            r
        });
        have_bound_regions
    }

    /// Folds the escaping and free regions in `value` using `f`, and
    /// sets `skipped_regions` to true if any late-bound region was found
    /// and skipped.
    pub fn fold_regions<T>(
        self,
        value: &T,
        skipped_regions: &mut bool,
        mut f: impl FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> T
    where
        T : TypeFoldable<'tcx>,
    {
        let Ok(r) = value.fold_with(&mut RegionFolder::new(self, skipped_regions, &mut f));
        r
    }

    /// Invoke `callback` on every region appearing free in `value`.
    pub fn for_each_free_region(
        self,
        value: &impl TypeFoldable<'tcx>,
        mut callback: impl FnMut(ty::Region<'tcx>),
    ) {
        self.any_free_region_meets(value, |r| {
            callback(r);
            false
        });
    }

    /// Returns `true` if `callback` returns true for every region appearing free in `value`.
    pub fn all_free_regions_meet(
        self,
        value: &impl TypeFoldable<'tcx>,
        mut callback: impl FnMut(ty::Region<'tcx>) -> bool,
    ) -> bool {
        !self.any_free_region_meets(value, |r| !callback(r))
    }

    /// Returns `true` if `callback` returns true for some region appearing free in `value`.
    pub fn any_free_region_meets(
        self,
        value: &impl TypeFoldable<'tcx>,
        callback: impl FnMut(ty::Region<'tcx>) -> bool,
    ) -> bool {
        return value.visit_with(&mut RegionVisitor {
            outer_index: ty::INNERMOST,
            callback
        }).is_err();

        struct RegionVisitor<F> {
            /// The index of a binder *just outside* the things we have
            /// traversed. If we encounter a bound region bound by this
            /// binder or one outer to it, it appears free. Example:
            ///
            /// ```
            ///    for<'a> fn(for<'b> fn(), T)
            /// ^          ^          ^     ^
            /// |          |          |     | here, would be shifted in 1
            /// |          |          | here, would be shifted in 2
            /// |          | here, would be `INNERMOST` shifted in by 1
            /// | here, initially, binder would be `INNERMOST`
            /// ```
            ///
            /// You see that, initially, *any* bound value is free,
            /// because we've not traversed any binders. As we pass
            /// through a binder, we shift the `outer_index` by 1 to
            /// account for the new binder that encloses us.
            outer_index: ty::DebruijnIndex,
            callback: F,
        }

        impl<'tcx, F> TypeVisitor<'tcx> for RegionVisitor<F>
            where F: FnMut(ty::Region<'tcx>) -> bool
        {
            type Error = ();

            fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> Result<(), ()> {
                self.outer_index.shift_in(1);
                let result = t.skip_binder().visit_with(self);
                self.outer_index.shift_out(1);
                result
            }

            fn visit_region(&mut self, r: ty::Region<'tcx>) -> Result<(), ()> {
                match *r {
                    ty::ReLateBound(debruijn, _) if debruijn < self.outer_index => {
                        Ok(()) // ignore bound regions, keep visiting
                    }
                    _ => match (self.callback)(r) {
                        true => Err(()),
                        false => Ok(())
                    }
                }
            }

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Result<(), ()> {
                // We're only interested in types involving regions
                if ty.flags.intersects(TypeFlags::HAS_FREE_REGIONS) {
                    ty.super_visit_with(self)
                } else {
                    Ok(()) // keep visiting
                }
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

    /// Stores the index of a binder *just outside* the stuff we have
    /// visited.  So this begins as INNERMOST; when we pass through a
    /// binder, it is incremented (via `shift_in`).
    current_index: ty::DebruijnIndex,

    /// Callback invokes for each free region. The `DebruijnIndex`
    /// points to the binder *just outside* the ones we have passed
    /// through.
    fold_region_fn: &'a mut (dyn FnMut(
        ty::Region<'tcx>,
        ty::DebruijnIndex,
    ) -> ty::Region<'tcx> + 'a),
}

impl<'a, 'gcx, 'tcx> RegionFolder<'a, 'gcx, 'tcx> {
    #[inline]
    pub fn new(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        skipped_regions: &'a mut bool,
        fold_region_fn: &'a mut dyn FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> RegionFolder<'a, 'gcx, 'tcx> {
        RegionFolder {
            tcx,
            skipped_regions,
            current_index: ty::INNERMOST,
            fold_region_fn,
        }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionFolder<'a, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>)
                                          -> Result<ty::Binder<T>, !>
    {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        match *r {
            ty::ReLateBound(debruijn, _) if debruijn < self.current_index => {
                debug!("RegionFolder.fold_region({:?}) skipped bound region (current index={:?})",
                       r, self.current_index);
                *self.skipped_regions = true;
                Ok(r)
            }
            _ => {
                debug!("RegionFolder.fold_region({:?}) folding free region (current_index={:?})",
                       r, self.current_index);
                Ok((self.fold_region_fn)(r, self.current_index))
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Bound vars replacer

/// Replaces the escaping bound vars (late bound regions or bound types) in a type.
struct BoundVarReplacer<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,

    /// As with `RegionFolder`, represents the index of a binder *just outside*
    /// the ones we have visited.
    current_index: ty::DebruijnIndex,

    fld_r: &'a mut (dyn FnMut(ty::BoundRegion) -> ty::Region<'tcx> + 'a),
    fld_t: &'a mut (dyn FnMut(ty::BoundTy) -> ty::Ty<'tcx> + 'a),
}

impl<'a, 'gcx, 'tcx> BoundVarReplacer<'a, 'gcx, 'tcx> {
    fn new<F, G>(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        fld_r: &'a mut F,
        fld_t: &'a mut G
    ) -> Self
        where F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
              G: FnMut(ty::BoundTy) -> ty::Ty<'tcx>
    {
        BoundVarReplacer {
            tcx,
            current_index: ty::INNERMOST,
            fld_r,
            fld_t,
        }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for BoundVarReplacer<'a, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>)
                                          -> Result<ty::Binder<T>, !>
    {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        Ok(match t.sty {
            ty::Bound(debruijn, bound_ty) => {
                if debruijn == self.current_index {
                    let fld_t = &mut self.fld_t;
                    let ty = fld_t(bound_ty);
                    ty::fold::shift_vars(
                        self.tcx,
                        &ty,
                        self.current_index.as_u32()
                    )
                } else {
                    t
                }
            }
            _ => {
                if !t.has_vars_bound_at_or_above(self.current_index) {
                    // Nothing more to substitute.
                    t
                } else {
                    t.super_fold_with(self)?
                }
            }
        })
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        Ok(match *r {
            ty::ReLateBound(debruijn, br) if debruijn == self.current_index => {
                let fld_r = &mut self.fld_r;
                let region = fld_r(br);
                if let ty::ReLateBound(debruijn1, br) = *region {
                    // If the callback returns a late-bound region,
                    // that region should always use the INNERMOST
                    // debruijn index. Then we adjust it to the
                    // correct depth.
                    assert_eq!(debruijn1, ty::INNERMOST);
                    self.tcx.mk_region(ty::ReLateBound(debruijn, br))
                } else {
                    region
                }
            }
            _ => r
        })
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    /// Replaces all regions bound by the given `Binder` with the
    /// results returned by the closure; the closure is expected to
    /// return a free region (relative to this binder), and hence the
    /// binder is removed in the return type. The closure is invoked
    /// once for each unique `BoundRegion`; multiple references to the
    /// same `BoundRegion` will reuse the previous result. A map is
    /// returned at the end with each bound region and the free region
    /// that replaced it.
    ///
    /// This method only replaces late bound regions and the result may still
    /// contain escaping bound types.
    pub fn replace_late_bound_regions<T, F>(
        self,
        value: &Binder<T>,
        fld_r: F
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
        where F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
              T: TypeFoldable<'tcx>
    {
        // identity for bound types
        let fld_t = |bound_ty| self.mk_ty(ty::Bound(ty::INNERMOST, bound_ty));
        self.replace_escaping_bound_vars(value.skip_binder(), fld_r, fld_t)
    }

    /// Replaces all escaping bound vars. The `fld_r` closure replaces escaping
    /// bound regions while the `fld_t` closure replaces escaping bound types.
    pub fn replace_escaping_bound_vars<T, F, G>(
        self,
        value: &T,
        mut fld_r: F,
        mut fld_t: G
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
        where F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
              G: FnMut(ty::BoundTy) -> ty::Ty<'tcx>,
              T: TypeFoldable<'tcx>
    {
        use rustc_data_structures::fx::FxHashMap;

        let mut region_map = BTreeMap::new();
        let mut type_map = FxHashMap::default();

        if !value.has_escaping_bound_vars() {
            (value.clone(), region_map)
        } else {
            let mut real_fld_r = |br| {
                *region_map.entry(br).or_insert_with(|| fld_r(br))
            };

            let mut real_fld_t = |bound_ty| {
                *type_map.entry(bound_ty).or_insert_with(|| fld_t(bound_ty))
            };

            let mut replacer = BoundVarReplacer::new(self, &mut real_fld_r, &mut real_fld_t);
            let Ok(result) = value.fold_with(&mut replacer);
            (result, region_map)
        }
    }

    /// Replaces all types or regions bound by the given `Binder`. The `fld_r`
    /// closure replaces bound regions while the `fld_t` closure replaces bound
    /// types.
    pub fn replace_bound_vars<T, F, G>(
        self,
        value: &Binder<T>,
        fld_r: F,
        fld_t: G
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
        where F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
              G: FnMut(ty::BoundTy) -> ty::Ty<'tcx>,
              T: TypeFoldable<'tcx>
    {
        self.replace_escaping_bound_vars(value.skip_binder(), fld_r, fld_t)
    }

    /// Replaces any late-bound regions bound in `value` with
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
        let Ok(_) = value.skip_binder().visit_with(&mut collector);
        collector.regions
    }

    /// Replaces any late-bound regions bound in `value` with `'erased`. Useful in codegen but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn erase_late_bound_regions<T>(self, value: &Binder<T>) -> T
        where T : TypeFoldable<'tcx>
    {
        self.replace_late_bound_regions(value, |_| self.types.re_erased).0
    }

    /// Rewrite any late-bound regions so that they are anonymous. Region numbers are
    /// assigned starting at 1 and increasing monotonically in the order traversed
    /// by the fold operation.
    ///
    /// The chief purpose of this function is to canonicalize regions so that two
    /// `FnSig`s or `TraitRef`s which are equivalent up to region naming will become
    /// structurally identical. For example, `for<'a, 'b> fn(&'a isize, &'b isize)` and
    /// `for<'a, 'b> fn(&'b isize, &'a isize)` will become identical after anonymization.
    pub fn anonymize_late_bound_regions<T>(self, sig: &Binder<T>) -> Binder<T>
        where T : TypeFoldable<'tcx>,
    {
        let mut counter = 0;
        Binder::bind(self.replace_late_bound_regions(sig, |_| {
            counter += 1;
            self.mk_region(ty::ReLateBound(ty::INNERMOST, ty::BrAnon(counter)))
        }).0)
    }
}

///////////////////////////////////////////////////////////////////////////
// Shifter
//
// Shifts the De Bruijn indices on all escaping bound vars by a
// fixed amount. Useful in substitution or when otherwise introducing
// a binding level that is not intended to capture the existing bound
// vars. See comment on `shift_vars_through_binders` method in
// `subst.rs` for more details.

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum Direction {
    In,
    Out,
}

struct Shifter<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    current_index: ty::DebruijnIndex,
    amount: u32,
    direction: Direction,
}

impl Shifter<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>, amount: u32, direction: Direction) -> Self {
        Shifter {
            tcx,
            current_index: ty::INNERMOST,
            amount,
            direction,
        }
    }
}

impl TypeFolder<'gcx, 'tcx> for Shifter<'a, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: &ty::Binder<T>)
                                          -> Result<ty::Binder<T>, !>
    {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        Ok(match *r {
            ty::ReLateBound(debruijn, br) => {
                if self.amount == 0 || debruijn < self.current_index {
                    r
                } else {
                    let debruijn = match self.direction {
                        Direction::In => debruijn.shifted_in(self.amount),
                        Direction::Out => {
                            assert!(debruijn.as_u32() >= self.amount);
                            debruijn.shifted_out(self.amount)
                        }
                    };
                    let shifted = ty::ReLateBound(debruijn, br);
                    self.tcx.mk_region(shifted)
                }
            }
            _ => r
        })
    }

    fn fold_ty(&mut self, ty: ty::Ty<'tcx>) -> Result<ty::Ty<'tcx>, !> {
        Ok(match ty.sty {
            ty::Bound(debruijn, bound_ty) => {
                if self.amount == 0 || debruijn < self.current_index {
                    ty
                } else {
                    let debruijn = match self.direction {
                        Direction::In => debruijn.shifted_in(self.amount),
                        Direction::Out => {
                            assert!(debruijn.as_u32() >= self.amount);
                            debruijn.shifted_out(self.amount)
                        }
                    };
                    self.tcx.mk_ty(
                        ty::Bound(debruijn, bound_ty)
                    )
                }
            }

            _ => ty.super_fold_with(self)?,
        })
    }
}

pub fn shift_region<'a, 'gcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    region: ty::Region<'tcx>,
    amount: u32
) -> ty::Region<'tcx> {
    match region {
        ty::ReLateBound(debruijn, br) if amount > 0 => {
            tcx.mk_region(ty::ReLateBound(debruijn.shifted_in(amount), *br))
        }
        _ => {
            region
        }
    }
}

pub fn shift_vars<'a, 'gcx, 'tcx, T>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    value: &T,
    amount: u32
) -> T where T: TypeFoldable<'tcx> {
    debug!("shift_vars(value={:?}, amount={})",
           value, amount);

    value.fold_with(&mut Shifter::new(tcx, amount, Direction::In)).unwrap_or_else(|e: !| e)
}

pub fn shift_out_vars<'a, 'gcx, 'tcx, T>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    value: &T,
    amount: u32
) -> T where T: TypeFoldable<'tcx> {
    debug!("shift_out_vars(value={:?}, amount={})",
           value, amount);

    value.fold_with(&mut Shifter::new(tcx, amount, Direction::Out)).unwrap_or_else(|e: !| e)
}

/// An "escaping var" is a bound var whose binder is not part of `t`. A bound var can be a
/// bound region or a bound type.
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
/// Note that what I'm calling an "escaping var" is often just called a "free var". However,
/// we already use the term "free var". It refers to the regions or types that we use to represent
/// bound regions or type params on a fn definition while we are type checking its body.
///
/// To clarify, conceptually there is no particular difference between
/// an "escaping" var and a "free" var. However, there is a big
/// difference in practice. Basically, when "entering" a binding
/// level, one is generally required to do some sort of processing to
/// a bound var, such as replacing it with a fresh/placeholder
/// var, or making an entry in the environment to represent the
/// scope to which it is attached, etc. An escaping var represents
/// a bound var for which this processing has not yet been done.
struct HasEscapingVarsVisitor {
    /// Anything bound by `outer_index` or "above" is escaping.
    outer_index: ty::DebruijnIndex,
}

impl<'tcx> TypeVisitor<'tcx> for HasEscapingVarsVisitor {
    type Error = ();

    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> Result<(), ()> {
        self.outer_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.outer_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Result<(), ()> {
        // If the outer-exclusive-binder is *strictly greater* than
        // `outer_index`, that means that `t` contains some content
        // bound at `outer_index` or above (because
        // `outer_exclusive_binder` is always 1 higher than the
        // content in `t`). Therefore, `t` has some escaping vars.
        match t.outer_exclusive_binder > self.outer_index {
            true => Err(()),
            false => Ok(())
        }
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> Result<(), ()> {
        // If the region is bound by `outer_index` or anything outside
        // of outer index, then it escapes the binders we have
        // visited.
        match r.bound_at_or_above_binder(self.outer_index) {
            true => Err(()),
            false => Ok(())
        }
    }
}

struct HasTypeFlagsVisitor {
    flags: ty::TypeFlags,
}

impl<'tcx> TypeVisitor<'tcx> for HasTypeFlagsVisitor {
    type Error = ();

    fn visit_ty(&mut self, t: Ty<'_>) -> Result<(), ()> {
        debug!("HasTypeFlagsVisitor: t={:?} t.flags={:?} self.flags={:?}", t, t.flags, self.flags);
        match t.flags.intersects(self.flags) {
            true => Err(()),
            false => Ok(())
        }
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> Result<(), ()> {
        let flags = r.type_flags();
        debug!("HasTypeFlagsVisitor: r={:?} r.flags={:?} self.flags={:?}", r, flags, self.flags);
        match flags.intersects(self.flags) {
            true => Err(()),
            false => Ok(())
        }
    }

    fn visit_const(&mut self, c: &'tcx ty::LazyConst<'tcx>) -> Result<(), ()> {
        if let ty::LazyConst::Unevaluated(..) = c {
            let projection_flags = TypeFlags::HAS_NORMALIZABLE_PROJECTION |
                TypeFlags::HAS_PROJECTION;
            if projection_flags.intersects(self.flags) {
                return Err(());
            }
        }
        c.super_visit_with(self)
    }
}

/// Collects all the late-bound regions at the innermost binding level
/// into a hash set.
struct LateBoundRegionsCollector {
    current_index: ty::DebruijnIndex,
    regions: FxHashSet<ty::BoundRegion>,

    /// `true` if we only want regions that are known to be
    /// "constrained" when you equate this type with another type. In
    /// particular, if you have e.g., `&'a u32` and `&'b u32`, equating
    /// them constraints `'a == 'b`. But if you have `<&'a u32 as
    /// Trait>::Foo` and `<&'b u32 as Trait>::Foo`, normalizing those
    /// types may mean that `'a` and `'b` don't appear in the results,
    /// so they are not considered *constrained*.
    just_constrained: bool,
}

impl LateBoundRegionsCollector {
    fn new(just_constrained: bool) -> Self {
        LateBoundRegionsCollector {
            current_index: ty::INNERMOST,
            regions: Default::default(),
            just_constrained,
        }
    }
}

impl<'tcx> TypeVisitor<'tcx> for LateBoundRegionsCollector {
    type Error = !;

    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> Result<(), !> {
        self.current_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.current_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Result<(), !> {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs to a projection, as they may not appear
        // in the normalized form
        if self.just_constrained {
            match t.sty {
                ty::Projection(..) | ty::Opaque(..) => { return Ok(()); }
                _ => { }
            }
        }

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> Result<(), !> {
        if let ty::ReLateBound(debruijn, br) = *r {
             if debruijn == self.current_index {
                self.regions.insert(br);
            }
        }
        Ok(())
    }
}
