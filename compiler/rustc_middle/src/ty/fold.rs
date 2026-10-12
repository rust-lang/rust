use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::DefId;
use rustc_span::bug;
use rustc_type_ir::{BoundVarReplacerDelegate, FnMutDelegate};

use crate::ty::{self, Binder, BoundTy, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable};

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
    fn cx(&self) -> TyCtxt<'tcx> {
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
    pub fn instantiate_bound_regions<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut fld_r: F,
    ) -> (T, FxIndexMap<ty::BoundRegion<'tcx>, ty::Region<'tcx>>)
    where
        F: FnMut(ty::BoundRegion<'tcx>) -> ty::Region<'tcx>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut region_map = FxIndexMap::default();
        let real_fld_r =
            |br: ty::BoundRegion<'tcx>| *region_map.entry(br).or_insert_with(|| fld_r(br));
        let value = self.instantiate_bound_regions_uncached(value, real_fld_r);
        (value, region_map)
    }

    pub fn instantiate_bound_regions_uncached<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut replace_regions: F,
    ) -> T
    where
        F: FnMut(ty::BoundRegion<'tcx>) -> ty::Region<'tcx>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let delegate = FnMutDelegate {
            regions: &mut replace_regions,
            types: &mut |b| bug!("unexpected bound ty in binder: {b:?}"),
            consts: &mut |b| bug!("unexpected bound ct in binder: {b:?}"),
        };
        rustc_type_ir::replace_escaping_bound_vars_uncached(self, value.skip_binder(), delegate)
    }

    /// Replaces escaping bound vars using `delegate`.
    pub fn replace_escaping_bound_vars_uncached<T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        value: T,
        delegate: impl BoundVarReplacerDelegate<TyCtxt<'tcx>>,
    ) -> T {
        rustc_type_ir::replace_escaping_bound_vars_uncached(self, value, delegate)
    }

    /// Replaces the variables bound by `value` using `delegate`.
    pub fn replace_bound_vars_uncached<T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        value: Binder<'tcx, T>,
        delegate: impl BoundVarReplacerDelegate<TyCtxt<'tcx>>,
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
        self.instantiate_bound_regions_uncached(value, |br| {
            let kind = ty::LateParamRegionKind::from_bound(br.var, br.kind);
            ty::Region::new_late_param(self, all_outlive_scope, kind)
        })
    }

    pub fn shift_bound_var_indices<T>(self, bound_vars: usize, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        rustc_type_ir::shift_bound_var_indices(self, bound_vars, value)
    }

    /// Replaces any late-bound regions bound in `value` with `'erased`. Useful in codegen but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn instantiate_bound_regions_with_erased<T>(self, value: Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.instantiate_bound_regions(value, |_| self.lifetimes.re_erased).0
    }

    /// Anonymize all bound variables in `value`, this is mostly used to improve caching.
    pub fn anonymize_bound_vars<T>(self, value: Binder<'tcx, T>) -> Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        struct Anonymize<'a, 'tcx> {
            tcx: TyCtxt<'tcx>,
            map: &'a mut FxIndexMap<ty::BoundVar, ty::BoundVariableKind<'tcx>>,
        }
        impl<'tcx> BoundVarReplacerDelegate<TyCtxt<'tcx>> for Anonymize<'_, 'tcx> {
            fn replace_region(&mut self, br: ty::BoundRegion<'tcx>) -> ty::Region<'tcx> {
                let entry = self.map.entry(br.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon))
                    .expect_region();
                let br = ty::BoundRegion { var, kind };
                ty::Region::new_bound(self.tcx, ty::INNERMOST, br)
            }
            fn replace_ty(&mut self, bt: ty::BoundTy<'tcx>) -> Ty<'tcx> {
                let entry = self.map.entry(bt.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon))
                    .expect_ty();
                Ty::new_bound(self.tcx, ty::INNERMOST, BoundTy { var, kind })
            }
            fn replace_const(&mut self, bc: ty::BoundConst<'tcx>) -> ty::Const<'tcx> {
                let entry = self.map.entry(bc.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let () = entry.or_insert_with(|| ty::BoundVariableKind::Const).expect_const();
                ty::Const::new_bound(self.tcx, ty::INNERMOST, ty::BoundConst::new(var))
            }
        }

        let mut map = Default::default();
        let delegate = Anonymize { tcx: self, map: &mut map };
        let inner = self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate);
        let bound_vars = self.mk_bound_variable_kinds_from_iter(map.into_values());
        Binder::bind_with_vars(inner, bound_vars)
    }
}
