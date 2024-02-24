use crate::ty::{self, Binder, Ty, TyCtxt, TypeFlags};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sso::SsoHashSet;
use rustc_type_ir::fold::TypeFoldable;
use std::ops::ControlFlow;

pub use rustc_type_ir::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};

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
            type BreakTy = ();

            fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
                &mut self,
                t: &Binder<'tcx, T>,
            ) -> ControlFlow<Self::BreakTy> {
                self.outer_index.shift_in(1);
                let result = t.super_visit_with(self);
                self.outer_index.shift_out(1);
                result
            }

            fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
                match *r {
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

            fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
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
    ) -> FxHashSet<ty::BoundRegionKind>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.collect_late_bound_regions(value, true)
    }

    /// Returns a set of all late-bound regions that appear in `value` anywhere.
    pub fn collect_referenced_late_bound_regions<T>(
        self,
        value: Binder<'tcx, T>,
    ) -> FxHashSet<ty::BoundRegionKind>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.collect_late_bound_regions(value, false)
    }

    fn collect_late_bound_regions<T>(
        self,
        value: Binder<'tcx, T>,
        just_constrained: bool,
    ) -> FxHashSet<ty::BoundRegionKind>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut collector = LateBoundRegionsCollector::new(just_constrained);
        let value = value.skip_binder();
        let value = if just_constrained { self.expand_weak_alias_tys(value) } else { value };
        let result = value.visit_with(&mut collector);
        assert!(result.is_continue()); // should never have stopped early
        collector.regions
    }
}

pub struct ValidateBoundVars<'tcx> {
    bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
    binder_index: ty::DebruijnIndex,
    // We may encounter the same variable at different levels of binding, so
    // this can't just be `Ty`
    visited: SsoHashSet<(ty::DebruijnIndex, Ty<'tcx>)>,
}

impl<'tcx> ValidateBoundVars<'tcx> {
    pub fn new(bound_vars: &'tcx ty::List<ty::BoundVariableKind>) -> Self {
        ValidateBoundVars {
            bound_vars,
            binder_index: ty::INNERMOST,
            visited: SsoHashSet::default(),
        }
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ValidateBoundVars<'tcx> {
    type BreakTy = ();

    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        t: &Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if t.outer_exclusive_binder() < self.binder_index
            || !self.visited.insert((self.binder_index, t))
        {
            return ControlFlow::Break(());
        }
        match *t.kind() {
            ty::Bound(debruijn, bound_ty) if debruijn == self.binder_index => {
                if self.bound_vars.len() <= bound_ty.var.as_usize() {
                    bug!("Not enough bound vars: {:?} not found in {:?}", t, self.bound_vars);
                }
                let list_var = self.bound_vars[bound_ty.var.as_usize()];
                match list_var {
                    ty::BoundVariableKind::Ty(kind) => {
                        if kind != bound_ty.kind {
                            bug!(
                                "Mismatched type kinds: {:?} doesn't var in list {:?}",
                                bound_ty.kind,
                                list_var
                            );
                        }
                    }
                    _ => {
                        bug!("Mismatched bound variable kinds! Expected type, found {:?}", list_var)
                    }
                }
            }

            _ => (),
        };

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *r {
            ty::ReBound(index, br) if index == self.binder_index => {
                if self.bound_vars.len() <= br.var.as_usize() {
                    bug!("Not enough bound vars: {:?} not found in {:?}", br, self.bound_vars);
                }
                let list_var = self.bound_vars[br.var.as_usize()];
                match list_var {
                    ty::BoundVariableKind::Region(kind) => {
                        if kind != br.kind {
                            bug!(
                                "Mismatched region kinds: {:?} doesn't match var ({:?}) in list ({:?})",
                                br.kind,
                                list_var,
                                self.bound_vars
                            );
                        }
                    }
                    _ => bug!(
                        "Mismatched bound variable kinds! Expected region, found {:?}",
                        list_var
                    ),
                }
            }

            _ => (),
        };

        ControlFlow::Continue(())
    }
}

/// Collects all the late-bound regions at the innermost binding level
/// into a hash set.
struct LateBoundRegionsCollector {
    current_index: ty::DebruijnIndex,
    regions: FxHashSet<ty::BoundRegionKind>,

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
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        t: &Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.current_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.current_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if self.just_constrained {
            match t.kind() {
                // If we are only looking for "constrained" regions, we have to ignore the
                // inputs to a projection as they may not appear in the normalized form.
                ty::Alias(ty::Projection | ty::Inherent | ty::Opaque, _) => {
                    return ControlFlow::Continue(());
                }
                // All weak alias types should've been expanded beforehand.
                ty::Alias(ty::Weak, _) => bug!("unexpected weak alias type"),
                _ => {}
            }
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        // if we are only looking for "constrained" region, we have to
        // ignore the inputs of an unevaluated const, as they may not appear
        // in the normalized form
        if self.just_constrained {
            if let ty::ConstKind::Unevaluated(..) = c.kind() {
                return ControlFlow::Continue(());
            }
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::ReBound(debruijn, br) = *r {
            if debruijn == self.current_index {
                self.regions.insert(br.kind);
            }
        }
        ControlFlow::Continue(())
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
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::Placeholder(placeholder) = t.kind() {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::consts::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::ConstKind::Placeholder(placeholder) = c.kind() {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        if let ty::RePlaceholder(placeholder) = *r {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        ControlFlow::Continue(())
    }
}
