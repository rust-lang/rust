use std::ops::ControlFlow;

pub use rustc_type_ir::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};

use crate::ty::{self, Binder, Ty, TyCtxt, TypeFlags};

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
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::consts::Const<'tcx>) {
        if let ty::ConstKind::Placeholder(placeholder) = c.kind() {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        if let ty::RePlaceholder(placeholder) = *r {
            self.max_universe = ty::UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }
    }
}
