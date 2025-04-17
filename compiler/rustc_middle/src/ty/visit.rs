use std::ops::ControlFlow;

use rustc_data_structures::fx::FxIndexSet;
use rustc_type_ir::TypeFoldable;

use crate::ty::{
    self, Binder, Ty, TyCtxt, TypeFlags, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};

///////////////////////////////////////////////////////////////////////////
// Region folder

impl<'tcx> TyCtxt<'tcx> {
    /// Invoke `callback` on every region appearing free in `value`.
    pub fn for_each_free_region(
        self,
        value: &impl TypeVisitable<TyCtxt<'tcx>>,
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
        value: &impl TypeVisitable<TyCtxt<'tcx>>,
        mut callback: impl FnMut(ty::Region<'tcx>) -> bool,
    ) -> bool {
        !self.any_free_region_meets(value, |r| !callback(r))
    }

    /// Returns `true` if `callback` returns true for some region appearing free in `value`.
    pub fn any_free_region_meets(
        self,
        value: &impl TypeVisitable<TyCtxt<'tcx>>,
        callback: impl FnMut(ty::Region<'tcx>) -> bool,
    ) -> bool {
        struct RegionVisitor<F> {
            /// The index of a binder *just outside* the things we have
            /// traversed. If we encounter a bound region bound by this
            /// binder or one outer to it, it appears free. Example:
            ///
            /// ```ignore (illustrative)
            ///       for<'a> fn(for<'b> fn(), T)
            /// // ^          ^          ^     ^
            /// // |          |          |     | here, would be shifted in 1
            /// // |          |          | here, would be shifted in 2
            /// // |          | here, would be `INNERMOST` shifted in by 1
            /// // | here, initially, binder would be `INNERMOST`
            /// ```
            ///
            /// You see that, initially, *any* bound value is free,
            /// because we've not traversed any binders. As we pass
            /// through a binder, we shift the `outer_index` by 1 to
            /// account for the new binder that encloses us.
            outer_index: ty::DebruijnIndex,
            callback: F,
        }

        impl<'tcx, F> TypeVisitor<TyCtxt<'tcx>> for RegionVisitor<F>
        where
            F: FnMut(ty::Region<'tcx>) -> bool,
        {
            type Result = ControlFlow<()>;

            fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
                &mut self,
                t: &Binder<'tcx, T>,
            ) -> Self::Result {
                self.outer_index.shift_in(1);
                let result = t.super_visit_with(self);
                self.outer_index.shift_out(1);
                result
            }

            fn visit_region(&mut self, r: ty::Region<'tcx>) -> Self::Result {
                match r.kind() {
                    ty::ReBound(debruijn, _) if debruijn < self.outer_index => {
                        ControlFlow::Continue(())
                    }
                    _ => {
                        if (self.callback)(r) {
                            ControlFlow::Break(())
                        } else {
                            ControlFlow::Continue(())
                        }
                    }
                }
            }

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
                // We're only interested in types involving regions
                if ty.flags().intersects(TypeFlags::HAS_FREE_REGIONS) {
                    ty.super_visit_with(self)
                } else {
                    ControlFlow::Continue(())
                }
            }
        }

        value.visit_with(&mut RegionVisitor { outer_index: ty::INNERMOST, callback }).is_break()
    }

    /// Returns a set of all late-bound regions that are constrained
    /// by `value`, meaning that if we instantiate those LBR with
    /// variables and equate `value` with something else, those
    /// variables will also be equated.
    pub fn collect_constrained_late_bound_regions<T>(
        self,
        value: Binder<'tcx, T>,
    ) -> FxIndexSet<ty::BoundRegionKind>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.collect_late_bound_regions(value, true)
    }

    /// Returns a set of all late-bound regions that appear in `value` anywhere.
    pub fn collect_referenced_late_bound_regions<T>(
        self,
        value: Binder<'tcx, T>,
    ) -> FxIndexSet<ty::BoundRegionKind>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.collect_late_bound_regions(value, false)
    }

    fn collect_late_bound_regions<T>(
        self,
        value: Binder<'tcx, T>,
        just_constrained: bool,
    ) -> FxIndexSet<ty::BoundRegionKind>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut collector = LateBoundRegionsCollector::new(just_constrained);
        let value = value.skip_binder();
        let value = if just_constrained { self.expand_weak_alias_tys(value) } else { value };
        value.visit_with(&mut collector);
        collector.regions
    }
}

/// Collects all the late-bound regions at the innermost binding level
/// into a hash set.
struct LateBoundRegionsCollector {
    current_index: ty::DebruijnIndex,
    regions: FxIndexSet<ty::BoundRegionKind>,

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
        Self { current_index: ty::INNERMOST, regions: Default::default(), just_constrained }
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for LateBoundRegionsCollector {
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(&mut self, t: &Binder<'tcx, T>) {
        self.current_index.shift_in(1);
        t.super_visit_with(self);
        self.current_index.shift_out(1);
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) {
        if self.just_constrained {
            match t.kind() {
                // If we are only looking for "constrained" regions, we have to ignore the
                // inputs to a projection as they may not appear in the normalized form.
                ty::Alias(ty::Projection | ty::Inherent | ty::Opaque, _) => {
                    return;
                }
                // All weak alias types should've been expanded beforehand.
                ty::Alias(ty::Weak, _) => bug!("unexpected weak alias type"),
                _ => {}
            }
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs of an unevaluated const, as they may not appear
        // in the normalized form
        if self.just_constrained {
            if let ty::ConstKind::Unevaluated(..) = c.kind() {
                return;
            }
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        if let ty::ReBound(debruijn, br) = r.kind() {
            if debruijn == self.current_index {
                self.regions.insert(br.kind);
            }
        }
    }
}

/// Finds the max universe present
pub struct MaxUniverse {
    max_universe: ty::UniverseIndex,
}

impl MaxUniverse {
    pub fn new() -> Self {
        MaxUniverse { max_universe: ty::UniverseIndex::ROOT }
    }

    pub fn max_universe(self) -> ty::UniverseIndex {
        self.max_universe
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for MaxUniverse {
    fn visit_ty(&mut self, t: Ty<'tcx>) {
        if let ty::Placeholder(placeholder) = t.kind() {
            self.max_universe = self.max_universe.max(placeholder.universe);
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::consts::Const<'tcx>) {
        if let ty::ConstKind::Placeholder(placeholder) = c.kind() {
            self.max_universe = self.max_universe.max(placeholder.universe);
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        if let ty::RePlaceholder(placeholder) = r.kind() {
            self.max_universe = self.max_universe.max(placeholder.universe);
        }
    }
}
