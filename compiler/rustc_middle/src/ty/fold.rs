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
use crate::ty::{self, flags::FlagComputation, Binder, Ty, TyCtxt, TypeFlags};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;

use rustc_data_structures::fx::FxHashSet;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::ControlFlow;

/// This trait is implemented for every type that can be folded.
/// Basically, every type that has a corresponding method in `TypeFolder`.
///
/// To implement this conveniently, use the derive macro located in librustc_macros.
pub trait TypeFoldable<'tcx>: fmt::Debug + Clone {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self;
    fn fold_with<F: TypeFolder<'tcx>>(self, folder: &mut F) -> Self {
        self.super_fold_with(folder)
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy>;
    fn visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.super_visit_with(visitor)
    }

    /// Returns `true` if `self` has any late-bound regions that are either
    /// bound by `binder` or bound by some binder outside of `binder`.
    /// If `binder` is `ty::INNERMOST`, this indicates whether
    /// there are any late-bound regions that appear free.
    fn has_vars_bound_at_or_above(&self, binder: ty::DebruijnIndex) -> bool {
        self.visit_with(&mut HasEscapingVarsVisitor { outer_index: binder }).is_break()
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
        self.visit_with(&mut HasTypeFlagsVisitor { flags }).break_value() == Some(FoundFlags)
    }
    fn has_projections(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_PROJECTION)
    }
    fn has_opaque_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_OPAQUE)
    }
    fn references_error(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_ERROR)
    }
    fn has_param_types_or_consts(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_PARAM | TypeFlags::HAS_CT_PARAM)
    }
    fn has_infer_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_INFER)
    }
    fn has_infer_types(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER)
    }
    fn has_infer_types_or_consts(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_TY_INFER | TypeFlags::HAS_CT_INFER)
    }
    fn needs_infer(&self) -> bool {
        self.has_type_flags(TypeFlags::NEEDS_INFER)
    }
    fn has_placeholders(&self) -> bool {
        self.has_type_flags(
            TypeFlags::HAS_RE_PLACEHOLDER
                | TypeFlags::HAS_TY_PLACEHOLDER
                | TypeFlags::HAS_CT_PLACEHOLDER,
        )
    }
    fn needs_subst(&self) -> bool {
        self.has_type_flags(TypeFlags::NEEDS_SUBST)
    }
    /// "Free" regions in this context means that it has any region
    /// that is not (a) erased or (b) late-bound.
    fn has_free_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    fn has_erased_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_ERASED)
    }

    /// True if there are any un-erased free regions.
    fn has_erasable_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_FREE_REGIONS)
    }

    /// Indicates whether this value references only 'global'
    /// generic parameters that are the same regardless of what fn we are
    /// in. This is used for caching.
    fn is_global(&self) -> bool {
        !self.has_type_flags(TypeFlags::HAS_FREE_LOCAL_NAMES)
    }

    /// True if there are any late-bound regions
    fn has_late_bound_regions(&self) -> bool {
        self.has_type_flags(TypeFlags::HAS_RE_LATE_BOUND)
    }

    /// Indicates whether this value still has parameters/placeholders/inference variables
    /// which could be replaced later, in a way that would change the results of `impl`
    /// specialization.
    fn still_further_specializable(&self) -> bool {
        self.has_type_flags(TypeFlags::STILL_FURTHER_SPECIALIZABLE)
    }
}

impl TypeFoldable<'tcx> for hir::Constness {
    fn super_fold_with<F: TypeFolder<'tcx>>(self, _: &mut F) -> Self {
        self
    }
    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, _: &mut V) -> ControlFlow<V::BreakTy> {
        ControlFlow::CONTINUE
    }
}

/// The `TypeFolder` trait defines the actual *folding*. There is a
/// method defined for every foldable type. Each of these has a
/// default implementation that does an "identity" fold. Within each
/// identity fold, it should invoke `foo.fold_with(self)` to fold each
/// sub-item.
pub trait TypeFolder<'tcx>: Sized {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx>;

    fn fold_binder<T>(&mut self, t: Binder<T>) -> Binder<T>
    where
        T: TypeFoldable<'tcx>,
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

pub trait TypeVisitor<'tcx>: Sized {
    type BreakTy = !;

    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> ControlFlow<Self::BreakTy> {
        t.super_visit_with(self)
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        r.super_visit_with(self)
    }

    fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        c.super_visit_with(self)
    }

    fn visit_predicate(&mut self, p: ty::Predicate<'tcx>) -> ControlFlow<Self::BreakTy> {
        p.super_visit_with(self)
    }
}

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(&'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx>,
{
    pub tcx: TyCtxt<'tcx>,
    pub ty_op: F,
    pub lt_op: G,
    pub ct_op: H,
}

impl<'tcx, F, G, H> TypeFolder<'tcx> for BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(&'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx>,
{
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t = ty.super_fold_with(self);
        (self.ty_op)(t)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        let r = r.super_fold_with(self);
        (self.lt_op)(r)
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        let ct = ct.super_fold_with(self);
        (self.ct_op)(ct)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

impl<'tcx> TyCtxt<'tcx> {
    /// Folds the escaping and free regions in `value` using `f`, and
    /// sets `skipped_regions` to true if any late-bound region was found
    /// and skipped.
    pub fn fold_regions<T>(
        self,
        value: T,
        skipped_regions: &mut bool,
        mut f: impl FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        value.fold_with(&mut RegionFolder::new(self, skipped_regions, &mut f))
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
        where
            F: FnMut(ty::Region<'tcx>) -> bool,
        {
            type BreakTy = ();

            fn visit_binder<T: TypeFoldable<'tcx>>(
                &mut self,
                t: &Binder<T>,
            ) -> ControlFlow<Self::BreakTy> {
                self.outer_index.shift_in(1);
                let result = t.as_ref().skip_binder().visit_with(self);
                self.outer_index.shift_out(1);
                result
            }

            fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
                match *r {
                    ty::ReLateBound(debruijn, _) if debruijn < self.outer_index => {
                        ControlFlow::CONTINUE
                    }
                    _ => {
                        if (self.callback)(r) {
                            ControlFlow::BREAK
                        } else {
                            ControlFlow::CONTINUE
                        }
                    }
                }
            }

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                // We're only interested in types involving regions
                if ty.flags().intersects(TypeFlags::HAS_FREE_REGIONS) {
                    ty.super_visit_with(self)
                } else {
                    ControlFlow::CONTINUE
                }
            }
        }

        value.visit_with(&mut RegionVisitor { outer_index: ty::INNERMOST, callback }).is_break()
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

pub struct RegionFolder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    skipped_regions: &'a mut bool,

    /// Stores the index of a binder *just outside* the stuff we have
    /// visited.  So this begins as INNERMOST; when we pass through a
    /// binder, it is incremented (via `shift_in`).
    current_index: ty::DebruijnIndex,

    /// Callback invokes for each free region. The `DebruijnIndex`
    /// points to the binder *just outside* the ones we have passed
    /// through.
    fold_region_fn:
        &'a mut (dyn FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx> + 'a),
}

impl<'a, 'tcx> RegionFolder<'a, 'tcx> {
    #[inline]
    pub fn new(
        tcx: TyCtxt<'tcx>,
        skipped_regions: &'a mut bool,
        fold_region_fn: &'a mut dyn FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> RegionFolder<'a, 'tcx> {
        RegionFolder { tcx, skipped_regions, current_index: ty::INNERMOST, fold_region_fn }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for RegionFolder<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: ty::Binder<T>) -> ty::Binder<T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, _) if debruijn < self.current_index => {
                debug!(
                    "RegionFolder.fold_region({:?}) skipped bound region (current index={:?})",
                    r, self.current_index
                );
                *self.skipped_regions = true;
                r
            }
            _ => {
                debug!(
                    "RegionFolder.fold_region({:?}) folding free region (current_index={:?})",
                    r, self.current_index
                );
                (self.fold_region_fn)(r, self.current_index)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Bound vars replacer

/// Replaces the escaping bound vars (late bound regions or bound types) in a type.
struct BoundVarReplacer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,

    /// As with `RegionFolder`, represents the index of a binder *just outside*
    /// the ones we have visited.
    current_index: ty::DebruijnIndex,

    fld_r: &'a mut (dyn FnMut(ty::BoundRegion) -> ty::Region<'tcx> + 'a),
    fld_t: &'a mut (dyn FnMut(ty::BoundTy) -> Ty<'tcx> + 'a),
    fld_c: &'a mut (dyn FnMut(ty::BoundVar, Ty<'tcx>) -> &'tcx ty::Const<'tcx> + 'a),
}

impl<'a, 'tcx> BoundVarReplacer<'a, 'tcx> {
    fn new<F, G, H>(tcx: TyCtxt<'tcx>, fld_r: &'a mut F, fld_t: &'a mut G, fld_c: &'a mut H) -> Self
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        G: FnMut(ty::BoundTy) -> Ty<'tcx>,
        H: FnMut(ty::BoundVar, Ty<'tcx>) -> &'tcx ty::Const<'tcx>,
    {
        BoundVarReplacer { tcx, current_index: ty::INNERMOST, fld_r, fld_t, fld_c }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for BoundVarReplacer<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: ty::Binder<T>) -> ty::Binder<T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Bound(debruijn, bound_ty) => {
                if debruijn == self.current_index {
                    let fld_t = &mut self.fld_t;
                    let ty = fld_t(bound_ty);
                    ty::fold::shift_vars(self.tcx, &ty, self.current_index.as_u32())
                } else {
                    t
                }
            }
            _ => {
                if !t.has_vars_bound_at_or_above(self.current_index) {
                    // Nothing more to substitute.
                    t
                } else {
                    t.super_fold_with(self)
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
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
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if let ty::Const { val: ty::ConstKind::Bound(debruijn, bound_const), ty } = *ct {
            if debruijn == self.current_index {
                let fld_c = &mut self.fld_c;
                let ct = fld_c(bound_const, ty);
                ty::fold::shift_vars(self.tcx, &ct, self.current_index.as_u32())
            } else {
                ct
            }
        } else {
            if !ct.has_vars_bound_at_or_above(self.current_index) {
                // Nothing more to substitute.
                ct
            } else {
                ct.super_fold_with(self)
            }
        }
    }
}

impl<'tcx> TyCtxt<'tcx> {
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
        value: Binder<T>,
        fld_r: F,
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        T: TypeFoldable<'tcx>,
    {
        // identity for bound types and consts
        let fld_t = |bound_ty| self.mk_ty(ty::Bound(ty::INNERMOST, bound_ty));
        let fld_c = |bound_ct, ty| {
            self.mk_const(ty::Const { val: ty::ConstKind::Bound(ty::INNERMOST, bound_ct), ty })
        };
        self.replace_escaping_bound_vars(value.skip_binder(), fld_r, fld_t, fld_c)
    }

    /// Replaces all escaping bound vars. The `fld_r` closure replaces escaping
    /// bound regions; the `fld_t` closure replaces escaping bound types and the `fld_c`
    /// closure replaces escaping bound consts.
    pub fn replace_escaping_bound_vars<T, F, G, H>(
        self,
        value: T,
        mut fld_r: F,
        mut fld_t: G,
        mut fld_c: H,
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        G: FnMut(ty::BoundTy) -> Ty<'tcx>,
        H: FnMut(ty::BoundVar, Ty<'tcx>) -> &'tcx ty::Const<'tcx>,
        T: TypeFoldable<'tcx>,
    {
        use rustc_data_structures::fx::FxHashMap;

        let mut region_map = BTreeMap::new();
        let mut type_map = FxHashMap::default();
        let mut const_map = FxHashMap::default();

        if !value.has_escaping_bound_vars() {
            (value, region_map)
        } else {
            let mut real_fld_r = |br| *region_map.entry(br).or_insert_with(|| fld_r(br));

            let mut real_fld_t =
                |bound_ty| *type_map.entry(bound_ty).or_insert_with(|| fld_t(bound_ty));

            let mut real_fld_c =
                |bound_ct, ty| *const_map.entry(bound_ct).or_insert_with(|| fld_c(bound_ct, ty));

            let mut replacer =
                BoundVarReplacer::new(self, &mut real_fld_r, &mut real_fld_t, &mut real_fld_c);
            let result = value.fold_with(&mut replacer);
            (result, region_map)
        }
    }

    /// Replaces all types or regions bound by the given `Binder`. The `fld_r`
    /// closure replaces bound regions while the `fld_t` closure replaces bound
    /// types.
    pub fn replace_bound_vars<T, F, G, H>(
        self,
        value: Binder<T>,
        fld_r: F,
        fld_t: G,
        fld_c: H,
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        G: FnMut(ty::BoundTy) -> Ty<'tcx>,
        H: FnMut(ty::BoundVar, Ty<'tcx>) -> &'tcx ty::Const<'tcx>,
        T: TypeFoldable<'tcx>,
    {
        self.replace_escaping_bound_vars(value.skip_binder(), fld_r, fld_t, fld_c)
    }

    /// Replaces any late-bound regions bound in `value` with
    /// free variants attached to `all_outlive_scope`.
    pub fn liberate_late_bound_regions<T>(self, all_outlive_scope: DefId, value: ty::Binder<T>) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.replace_late_bound_regions(value, |br| {
            self.mk_region(ty::ReFree(ty::FreeRegion {
                scope: all_outlive_scope,
                bound_region: br,
            }))
        })
        .0
    }

    /// Returns a set of all late-bound regions that are constrained
    /// by `value`, meaning that if we instantiate those LBR with
    /// variables and equate `value` with something else, those
    /// variables will also be equated.
    pub fn collect_constrained_late_bound_regions<T>(
        self,
        value: &Binder<T>,
    ) -> FxHashSet<ty::BoundRegion>
    where
        T: TypeFoldable<'tcx>,
    {
        self.collect_late_bound_regions(value, true)
    }

    /// Returns a set of all late-bound regions that appear in `value` anywhere.
    pub fn collect_referenced_late_bound_regions<T>(
        self,
        value: &Binder<T>,
    ) -> FxHashSet<ty::BoundRegion>
    where
        T: TypeFoldable<'tcx>,
    {
        self.collect_late_bound_regions(value, false)
    }

    fn collect_late_bound_regions<T>(
        self,
        value: &Binder<T>,
        just_constraint: bool,
    ) -> FxHashSet<ty::BoundRegion>
    where
        T: TypeFoldable<'tcx>,
    {
        let mut collector = LateBoundRegionsCollector::new(just_constraint);
        let result = value.as_ref().skip_binder().visit_with(&mut collector);
        assert!(result.is_continue()); // should never have stopped early
        collector.regions
    }

    /// Replaces any late-bound regions bound in `value` with `'erased`. Useful in codegen but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn erase_late_bound_regions<T>(self, value: Binder<T>) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.replace_late_bound_regions(value, |_| self.lifetimes.re_erased).0
    }

    /// Rewrite any late-bound regions so that they are anonymous. Region numbers are
    /// assigned starting at 0 and increasing monotonically in the order traversed
    /// by the fold operation.
    ///
    /// The chief purpose of this function is to canonicalize regions so that two
    /// `FnSig`s or `TraitRef`s which are equivalent up to region naming will become
    /// structurally identical. For example, `for<'a, 'b> fn(&'a isize, &'b isize)` and
    /// `for<'a, 'b> fn(&'b isize, &'a isize)` will become identical after anonymization.
    pub fn anonymize_late_bound_regions<T>(self, sig: Binder<T>) -> Binder<T>
    where
        T: TypeFoldable<'tcx>,
    {
        let mut counter = 0;
        Binder::bind(
            self.replace_late_bound_regions(sig, |_| {
                let r = self.mk_region(ty::ReLateBound(ty::INNERMOST, ty::BrAnon(counter)));
                counter += 1;
                r
            })
            .0,
        )
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

struct Shifter<'tcx> {
    tcx: TyCtxt<'tcx>,
    current_index: ty::DebruijnIndex,
    amount: u32,
}

impl Shifter<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, amount: u32) -> Self {
        Shifter { tcx, current_index: ty::INNERMOST, amount }
    }
}

impl TypeFolder<'tcx> for Shifter<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(&mut self, t: ty::Binder<T>) -> ty::Binder<T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, br) => {
                if self.amount == 0 || debruijn < self.current_index {
                    r
                } else {
                    let debruijn = debruijn.shifted_in(self.amount);
                    let shifted = ty::ReLateBound(debruijn, br);
                    self.tcx.mk_region(shifted)
                }
            }
            _ => r,
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Bound(debruijn, bound_ty) => {
                if self.amount == 0 || debruijn < self.current_index {
                    ty
                } else {
                    let debruijn = debruijn.shifted_in(self.amount);
                    self.tcx.mk_ty(ty::Bound(debruijn, bound_ty))
                }
            }

            _ => ty.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if let ty::Const { val: ty::ConstKind::Bound(debruijn, bound_ct), ty } = *ct {
            if self.amount == 0 || debruijn < self.current_index {
                ct
            } else {
                let debruijn = debruijn.shifted_in(self.amount);
                self.tcx.mk_const(ty::Const { val: ty::ConstKind::Bound(debruijn, bound_ct), ty })
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}

pub fn shift_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    region: ty::Region<'tcx>,
    amount: u32,
) -> ty::Region<'tcx> {
    match region {
        ty::ReLateBound(debruijn, br) if amount > 0 => {
            tcx.mk_region(ty::ReLateBound(debruijn.shifted_in(amount), *br))
        }
        _ => region,
    }
}

pub fn shift_vars<'tcx, T>(tcx: TyCtxt<'tcx>, value: T, amount: u32) -> T
where
    T: TypeFoldable<'tcx>,
{
    debug!("shift_vars(value={:?}, amount={})", value, amount);

    value.fold_with(&mut Shifter::new(tcx, amount))
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct FoundEscapingVars;

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
    type BreakTy = FoundEscapingVars;

    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> ControlFlow<Self::BreakTy> {
        self.outer_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.outer_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        // If the outer-exclusive-binder is *strictly greater* than
        // `outer_index`, that means that `t` contains some content
        // bound at `outer_index` or above (because
        // `outer_exclusive_binder` is always 1 higher than the
        // content in `t`). Therefore, `t` has some escaping vars.
        if t.outer_exclusive_binder > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::CONTINUE
        }
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        // If the region is bound by `outer_index` or anything outside
        // of outer index, then it escapes the binders we have
        // visited.
        if r.bound_at_or_above_binder(self.outer_index) {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::CONTINUE
        }
    }

    fn visit_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        // we don't have a `visit_infer_const` callback, so we have to
        // hook in here to catch this case (annoying...), but
        // otherwise we do want to remember to visit the rest of the
        // const, as it has types/regions embedded in a lot of other
        // places.
        match ct.val {
            ty::ConstKind::Bound(debruijn, _) if debruijn >= self.outer_index => {
                ControlFlow::Break(FoundEscapingVars)
            }
            _ => ct.super_visit_with(self),
        }
    }

    fn visit_predicate(&mut self, predicate: ty::Predicate<'tcx>) -> ControlFlow<Self::BreakTy> {
        if predicate.inner.outer_exclusive_binder > self.outer_index {
            ControlFlow::Break(FoundEscapingVars)
        } else {
            ControlFlow::CONTINUE
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct FoundFlags;

// FIXME: Optimize for checking for infer flags
struct HasTypeFlagsVisitor {
    flags: ty::TypeFlags,
}

impl<'tcx> TypeVisitor<'tcx> for HasTypeFlagsVisitor {
    type BreakTy = FoundFlags;

    fn visit_ty(&mut self, t: Ty<'_>) -> ControlFlow<Self::BreakTy> {
        debug!(
            "HasTypeFlagsVisitor: t={:?} t.flags={:?} self.flags={:?}",
            t,
            t.flags(),
            self.flags
        );
        if t.flags().intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        let flags = r.type_flags();
        debug!("HasTypeFlagsVisitor: r={:?} r.flags={:?} self.flags={:?}", r, flags, self.flags);
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }

    fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        let flags = FlagComputation::for_const(c);
        debug!("HasTypeFlagsVisitor: c={:?} c.flags={:?} self.flags={:?}", c, flags, self.flags);
        if flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
    }

    fn visit_predicate(&mut self, predicate: ty::Predicate<'tcx>) -> ControlFlow<Self::BreakTy> {
        debug!(
            "HasTypeFlagsVisitor: predicate={:?} predicate.flags={:?} self.flags={:?}",
            predicate, predicate.inner.flags, self.flags
        );
        if predicate.inner.flags.intersects(self.flags) {
            ControlFlow::Break(FoundFlags)
        } else {
            ControlFlow::CONTINUE
        }
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
    fn visit_binder<T: TypeFoldable<'tcx>>(&mut self, t: &Binder<T>) -> ControlFlow<Self::BreakTy> {
        self.current_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.current_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs to a projection, as they may not appear
        // in the normalized form
        if self.just_constrained {
            if let ty::Projection(..) | ty::Opaque(..) = t.kind() {
                return ControlFlow::CONTINUE;
            }
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: &'tcx ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs of an unevaluated const, as they may not appear
        // in the normalized form
        if self.just_constrained {
            if let ty::ConstKind::Unevaluated(..) = c.val {
                return ControlFlow::CONTINUE;
            }
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::ReLateBound(debruijn, br) = *r {
            if debruijn == self.current_index {
                self.regions.insert(br);
            }
        }
        ControlFlow::CONTINUE
    }
}
