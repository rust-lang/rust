use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};

use crate::infer::outlives::test_type_match;
use crate::infer::region_constraints::VerifyIfEq;

/// Visits free regions in the type that are relevant for liveness computation.
/// These regions are passed to `OP`.
///
/// Specifically, we visit all of the regions of types recursively, except if
/// the type is an alias, we look at the outlives bounds in the param-env
/// and alias's item bounds. If there is a unique outlives bound, then visit
/// that instead. If there is not a unique but there is a `'static` outlives
/// bound, then don't visit anything. Otherwise, walk through the opaque's
/// regions structurally.
pub struct FreeRegionsVisitor<'tcx, OP: FnMut(ty::Region<'tcx>)> {
    pub tcx: TyCtxt<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
    pub op: OP,
}

impl<'tcx, OP> TypeVisitor<TyCtxt<'tcx>> for FreeRegionsVisitor<'tcx, OP>
where
    OP: FnMut(ty::Region<'tcx>),
{
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        match r.kind() {
            // ignore bound regions, keep visiting
            ty::ReBound(_, _) => {}
            _ => (self.op)(r),
        }
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        // We're only interested in types involving regions
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return;
        }

        // FIXME: Don't consider alias bounds on types that have escaping bound
        // vars. See #117455.
        if ty.has_escaping_bound_vars() {
            return ty.super_visit_with(self);
        }

        match *ty.kind() {
            // We can prove that an alias is live two ways:
            // 1. All the components are live.
            //
            // 2. There is a known outlives bound or where-clause, and that
            //    region is live.
            //
            // We search through the item bounds and where clauses for
            // either `'static` or a unique outlives region, and if one is
            // found, we just need to prove that that region is still live.
            // If one is not found, then we continue to walk through the alias.
            ty::Alias(kind, ty::AliasTy { def_id, args, .. }) => {
                let tcx = self.tcx;
                let param_env = self.param_env;
                let outlives_bounds: Vec<_> = tcx
                    .item_bounds(def_id)
                    .iter_instantiated(tcx, args)
                    .chain(param_env.caller_bounds())
                    .filter_map(|clause| {
                        let outlives = clause.as_type_outlives_clause()?;
                        if let Some(outlives) = outlives.no_bound_vars()
                            && outlives.0 == ty
                        {
                            Some(outlives.1)
                        } else {
                            test_type_match::extract_verify_if_eq(
                                tcx,
                                &outlives.map_bound(|ty::OutlivesPredicate(ty, bound)| {
                                    VerifyIfEq { ty, bound }
                                }),
                                ty,
                            )
                        }
                    })
                    .collect();
                // If we find `'static`, then we know the alias doesn't capture *any* regions.
                // Otherwise, all of the outlives regions should be equal -- if they're not,
                // we don't really know how to proceed, so we continue recursing through the
                // alias.
                if outlives_bounds.contains(&tcx.lifetimes.re_static) {
                    // no
                } else if let Some(r) = outlives_bounds.first()
                    && outlives_bounds[1..].iter().all(|other_r| other_r == r)
                {
                    assert!(r.type_flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS));
                    r.visit_with(self);
                } else {
                    // Skip lifetime parameters that are not captured, since they do
                    // not need to be live.
                    let variances = tcx.opt_alias_variances(kind, def_id);

                    for (idx, s) in args.iter().enumerate() {
                        if variances.map(|variances| variances[idx]) != Some(ty::Bivariant) {
                            s.visit_with(self);
                        }
                    }
                }
            }

            _ => ty.super_visit_with(self),
        }
    }
}
