use crate::ty::{self, Binder, BoundTy, Ty, TyCtxt, TypeVisitableExt};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::DefId;

use std::collections::BTreeMap;

pub use rustc_type_ir::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(ty::Const<'tcx>) -> ty::Const<'tcx>,
{
    pub tcx: TyCtxt<'tcx>,
    pub ty_op: F,
    pub lt_op: G,
    pub ct_op: H,
}

impl<'tcx, F, G, H> TypeFolder<TyCtxt<'tcx>> for BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(ty::Const<'tcx>) -> ty::Const<'tcx>,
{
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t = ty.super_fold_with(self);
        (self.ty_op)(t)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        // This one is a little different, because `super_fold_with` is not
        // implemented on non-recursive `Region`.
        (self.lt_op)(r)
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let ct = ct.super_fold_with(self);
        (self.ct_op)(ct)
    }
}

///////////////////////////////////////////////////////////////////////////
// Region folder

impl<'tcx> TyCtxt<'tcx> {
    /// Folds the escaping and free regions in `value` using `f`.
    pub fn fold_regions<T>(
        self,
        value: T,
        mut f: impl FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        value.fold_with(&mut RegionFolder::new(self, &mut f))
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

    /// Stores the index of a binder *just outside* the stuff we have
    /// visited. So this begins as INNERMOST; when we pass through a
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
        fold_region_fn: &'a mut dyn FnMut(ty::Region<'tcx>, ty::DebruijnIndex) -> ty::Region<'tcx>,
    ) -> RegionFolder<'a, 'tcx> {
        RegionFolder { tcx, current_index: ty::INNERMOST, fold_region_fn }
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for RegionFolder<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    #[instrument(skip(self), level = "debug", ret)]
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, _) if debruijn < self.current_index => {
                debug!(?self.current_index, "skipped bound region");
                r
            }
            _ => {
                debug!(?self.current_index, "folding free region");
                (self.fold_region_fn)(r, self.current_index)
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Bound vars replacer

pub trait BoundVarReplacerDelegate<'tcx> {
    fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx>;
    fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx>;
    fn replace_const(&mut self, bv: ty::BoundVar, ty: Ty<'tcx>) -> ty::Const<'tcx>;
}

pub struct FnMutDelegate<'a, 'tcx> {
    pub regions: &'a mut (dyn FnMut(ty::BoundRegion) -> ty::Region<'tcx> + 'a),
    pub types: &'a mut (dyn FnMut(ty::BoundTy) -> Ty<'tcx> + 'a),
    pub consts: &'a mut (dyn FnMut(ty::BoundVar, Ty<'tcx>) -> ty::Const<'tcx> + 'a),
}

impl<'a, 'tcx> BoundVarReplacerDelegate<'tcx> for FnMutDelegate<'a, 'tcx> {
    fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx> {
        (self.regions)(br)
    }
    fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx> {
        (self.types)(bt)
    }
    fn replace_const(&mut self, bv: ty::BoundVar, ty: Ty<'tcx>) -> ty::Const<'tcx> {
        (self.consts)(bv, ty)
    }
}

/// Replaces the escaping bound vars (late bound regions or bound types) in a type.
struct BoundVarReplacer<'tcx, D> {
    tcx: TyCtxt<'tcx>,

    /// As with `RegionFolder`, represents the index of a binder *just outside*
    /// the ones we have visited.
    current_index: ty::DebruijnIndex,

    delegate: D,
}

impl<'tcx, D: BoundVarReplacerDelegate<'tcx>> BoundVarReplacer<'tcx, D> {
    fn new(tcx: TyCtxt<'tcx>, delegate: D) -> Self {
        BoundVarReplacer { tcx, current_index: ty::INNERMOST, delegate }
    }
}

impl<'tcx, D> TypeFolder<TyCtxt<'tcx>> for BoundVarReplacer<'tcx, D>
where
    D: BoundVarReplacerDelegate<'tcx>,
{
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Bound(debruijn, bound_ty) if debruijn == self.current_index => {
                let ty = self.delegate.replace_ty(bound_ty);
                debug_assert!(!ty.has_vars_bound_above(ty::INNERMOST));
                ty::fold::shift_vars(self.tcx, ty, self.current_index.as_u32())
            }
            _ if t.has_vars_bound_at_or_above(self.current_index) => t.super_fold_with(self),
            _ => t,
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, br) if debruijn == self.current_index => {
                let region = self.delegate.replace_region(br);
                if let ty::ReLateBound(debruijn1, br) = *region {
                    // If the callback returns a late-bound region,
                    // that region should always use the INNERMOST
                    // debruijn index. Then we adjust it to the
                    // correct depth.
                    assert_eq!(debruijn1, ty::INNERMOST);
                    ty::Region::new_late_bound(self.tcx, debruijn, br)
                } else {
                    region
                }
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Bound(debruijn, bound_const) if debruijn == self.current_index => {
                let ct = self.delegate.replace_const(bound_const, ct.ty());
                debug_assert!(!ct.has_vars_bound_above(ty::INNERMOST));
                ty::fold::shift_vars(self.tcx, ct, self.current_index.as_u32())
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Replaces all regions bound by the given `Binder` with the
    /// results returned by the closure; the closure is expected to
    /// return a free region (relative to this binder), and hence the
    /// binder is removed in the return type. The closure is invoked
    /// once for each unique `BoundRegionKind`; multiple references to the
    /// same `BoundRegionKind` will reuse the previous result. A map is
    /// returned at the end with each bound region and the free region
    /// that replaced it.
    ///
    /// # Panics
    ///
    /// This method only replaces late bound regions. Any types or
    /// constants bound by `value` will cause an ICE.
    pub fn replace_late_bound_regions<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut fld_r: F,
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut region_map = BTreeMap::new();
        let real_fld_r = |br: ty::BoundRegion| *region_map.entry(br).or_insert_with(|| fld_r(br));
        let value = self.replace_late_bound_regions_uncached(value, real_fld_r);
        (value, region_map)
    }

    pub fn replace_late_bound_regions_uncached<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut replace_regions: F,
    ) -> T
    where
        F: FnMut(ty::BoundRegion) -> ty::Region<'tcx>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let value = value.skip_binder();
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let delegate = FnMutDelegate {
                regions: &mut replace_regions,
                types: &mut |b| bug!("unexpected bound ty in binder: {b:?}"),
                consts: &mut |b, ty| bug!("unexpected bound ct in binder: {b:?} {ty}"),
            };
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }

    /// Replaces all escaping bound vars. The `fld_r` closure replaces escaping
    /// bound regions; the `fld_t` closure replaces escaping bound types and the `fld_c`
    /// closure replaces escaping bound consts.
    pub fn replace_escaping_bound_vars_uncached<T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        value: T,
        delegate: impl BoundVarReplacerDelegate<'tcx>,
    ) -> T {
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }

    /// Replaces all types or regions bound by the given `Binder`. The `fld_r`
    /// closure replaces bound regions, the `fld_t` closure replaces bound
    /// types, and `fld_c` replaces bound constants.
    pub fn replace_bound_vars_uncached<T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        value: Binder<'tcx, T>,
        delegate: impl BoundVarReplacerDelegate<'tcx>,
    ) -> T {
        self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate)
    }

    /// Replaces any late-bound regions bound in `value` with
    /// free variants attached to `all_outlive_scope`.
    pub fn liberate_late_bound_regions<T>(
        self,
        all_outlive_scope: DefId,
        value: ty::Binder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.replace_late_bound_regions_uncached(value, |br| {
            ty::Region::new_free(self, all_outlive_scope, br.kind)
        })
    }

    pub fn shift_bound_var_indices<T>(self, bound_vars: usize, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let shift_bv = |bv: ty::BoundVar| ty::BoundVar::from_usize(bv.as_usize() + bound_vars);
        self.replace_escaping_bound_vars_uncached(
            value,
            FnMutDelegate {
                regions: &mut |r: ty::BoundRegion| {
                    ty::Region::new_late_bound(
                        self,
                        ty::INNERMOST,
                        ty::BoundRegion { var: shift_bv(r.var), kind: r.kind },
                    )
                },
                types: &mut |t: ty::BoundTy| {
                    self.mk_bound(ty::INNERMOST, ty::BoundTy { var: shift_bv(t.var), kind: t.kind })
                },
                consts: &mut |c, ty: Ty<'tcx>| {
                    ty::Const::new_bound(self, ty::INNERMOST, shift_bv(c), ty)
                },
            },
        )
    }

    /// Replaces any late-bound regions bound in `value` with `'erased`. Useful in codegen but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn erase_late_bound_regions<T>(self, value: Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.replace_late_bound_regions(value, |_| self.lifetimes.re_erased).0
    }

    /// Anonymize all bound variables in `value`, this is mostly used to improve caching.
    pub fn anonymize_bound_vars<T>(self, value: Binder<'tcx, T>) -> Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        struct Anonymize<'a, 'tcx> {
            tcx: TyCtxt<'tcx>,
            map: &'a mut FxIndexMap<ty::BoundVar, ty::BoundVariableKind>,
        }
        impl<'tcx> BoundVarReplacerDelegate<'tcx> for Anonymize<'_, 'tcx> {
            fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx> {
                let entry = self.map.entry(br.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Region(ty::BrAnon(None)))
                    .expect_region();
                let br = ty::BoundRegion { var, kind };
                ty::Region::new_late_bound(self.tcx, ty::INNERMOST, br)
            }
            fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx> {
                let entry = self.map.entry(bt.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon))
                    .expect_ty();
                self.tcx.mk_bound(ty::INNERMOST, BoundTy { var, kind })
            }
            fn replace_const(&mut self, bv: ty::BoundVar, ty: Ty<'tcx>) -> ty::Const<'tcx> {
                let entry = self.map.entry(bv);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let () = entry.or_insert_with(|| ty::BoundVariableKind::Const).expect_const();
                ty::Const::new_bound(self.tcx, ty::INNERMOST, var, ty)
            }
        }

        let mut map = Default::default();
        let delegate = Anonymize { tcx: self, map: &mut map };
        let inner = self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate);
        let bound_vars = self.mk_bound_variable_kinds_from_iter(map.into_values());
        Binder::bind_with_vars(inner, bound_vars)
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

impl<'tcx> Shifter<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, amount: u32) -> Self {
        Shifter { tcx, current_index: ty::INNERMOST, amount }
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for Shifter<'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(debruijn, br) if debruijn >= self.current_index => {
                let debruijn = debruijn.shifted_in(self.amount);
                ty::Region::new_late_bound(self.tcx, debruijn, br)
            }
            _ => r,
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Bound(debruijn, bound_ty) if debruijn >= self.current_index => {
                let debruijn = debruijn.shifted_in(self.amount);
                self.tcx.mk_bound(debruijn, bound_ty)
            }

            _ if ty.has_vars_bound_at_or_above(self.current_index) => ty.super_fold_with(self),
            _ => ty,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Bound(debruijn, bound_ct) = ct.kind()
            && debruijn >= self.current_index
        {
            let debruijn = debruijn.shifted_in(self.amount);
            ty::Const::new_bound(self.tcx, debruijn, bound_ct, ct.ty())
        } else {
            ct.super_fold_with(self)
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }
}

pub fn shift_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    region: ty::Region<'tcx>,
    amount: u32,
) -> ty::Region<'tcx> {
    match *region {
        ty::ReLateBound(debruijn, br) if amount > 0 => {
            ty::Region::new_late_bound(tcx, debruijn.shifted_in(amount), br)
        }
        _ => region,
    }
}

pub fn shift_vars<'tcx, T>(tcx: TyCtxt<'tcx>, value: T, amount: u32) -> T
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    debug!("shift_vars(value={:?}, amount={})", value, amount);

    if amount == 0 || !value.has_escaping_bound_vars() {
        return value;
    }

    value.fold_with(&mut Shifter::new(tcx, amount))
}
