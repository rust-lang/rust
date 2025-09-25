use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::bug;
use rustc_middle::ty::{
    self, GenericArgsRef, Region, RegionVid, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable,
    TypeVisitor,
};
use tracing::{debug, instrument};

use super::DefiningUse;
use super::region_ctxt::RegionCtxt;
use crate::constraints::ConstraintSccIndex;

pub(super) fn apply_member_constraints<'tcx>(
    rcx: &mut RegionCtxt<'_, 'tcx>,
    defining_uses: &[DefiningUse<'tcx>],
) {
    // Start by collecting the member constraints of all defining uses.
    //
    // Applying member constraints can influence other member constraints,
    // so we first collect and then apply them.
    let mut member_constraints = Default::default();
    for defining_use in defining_uses {
        let mut visitor = CollectMemberConstraintsVisitor {
            rcx,
            defining_use,
            member_constraints: &mut member_constraints,
        };
        defining_use.hidden_type.ty.visit_with(&mut visitor);
    }

    // Now walk over the region graph, visiting the smallest regions first and then all
    // regions which have to outlive that one.
    //
    // Whenever we encounter a member region, we mutate the value of this SCC. This is
    // as if we'd introduce new outlives constraints. However, we discard these region
    // values after we've inferred the hidden types of opaques and apply the region
    // constraints by simply equating the actual hidden type with the inferred one.
    debug!(?member_constraints);
    for scc_a in rcx.constraint_sccs.all_sccs() {
        debug!(?scc_a);
        // Start by  adding the region values required by outlives constraints. This
        // matches how we compute the final region values in `fn compute_regions`.
        //
        // We need to do this here to get a lower bound when applying member constraints.
        // This propagates the region values added by previous member constraints.
        for &scc_b in rcx.constraint_sccs.successors(scc_a) {
            debug!(?scc_b);
            rcx.scc_values.add_region(scc_a, scc_b);
        }

        for defining_use in member_constraints.get(&scc_a).into_iter().flatten() {
            apply_member_constraint(rcx, scc_a, &defining_use.arg_regions);
        }
    }
}

#[instrument(level = "debug", skip(rcx))]
fn apply_member_constraint<'tcx>(
    rcx: &mut RegionCtxt<'_, 'tcx>,
    member: ConstraintSccIndex,
    arg_regions: &[RegionVid],
) {
    // If the member region lives in a higher universe, we currently choose
    // the most conservative option by leaving it unchanged.
    if !rcx.max_placeholder_universe_reached(member).is_root() {
        return;
    }

    // The existing value of `'member` is a lower-bound. If its is already larger than
    // some universal region, we cannot equate it with that region. Said differently, we
    // ignore choice regions which are smaller than this member region.
    let mut choice_regions = arg_regions
        .iter()
        .copied()
        .map(|r| rcx.representative(r).rvid())
        .filter(|&choice_region| {
            rcx.scc_values.universal_regions_outlived_by(member).all(|lower_bound| {
                rcx.universal_region_relations.outlives(choice_region, lower_bound)
            })
        })
        .collect::<Vec<_>>();
    debug!(?choice_regions, "after enforcing lower-bound");

    // Now find all the *upper bounds* -- that is, each UB is a
    // free region that must outlive the member region `R0` (`UB:
    // R0`). Therefore, we need only keep an option `O` if `UB: O`
    // for all UB.
    //
    // If we have a requirement `'upper_bound: 'member`, equating `'member`
    // with some region `'choice` means we now also require `'upper_bound: 'choice`.
    // Avoid choice regions for which this does not hold.
    for ub in rcx.rev_scc_graph.upper_bounds(member) {
        choice_regions
            .retain(|&choice_region| rcx.universal_region_relations.outlives(ub, choice_region));
    }
    debug!(?choice_regions, "after enforcing upper-bound");

    // At this point we can pick any member of `choice_regions` and would like to choose
    // it to be a small as possible. To avoid potential non-determinism we will pick the
    // smallest such choice.
    //
    // Because universal regions are only partially ordered (i.e, not every two regions are
    // comparable), we will ignore any region that doesn't compare to all others when picking
    // the minimum choice.
    //
    // For example, consider `choice_regions = ['static, 'a, 'b, 'c, 'd, 'e]`, where
    // `'static: 'a, 'static: 'b, 'a: 'c, 'b: 'c, 'c: 'd, 'c: 'e`.
    // `['d, 'e]` are ignored because they do not compare - the same goes for `['a, 'b]`.
    let totally_ordered_subset = choice_regions.iter().copied().filter(|&r1| {
        choice_regions.iter().all(|&r2| {
            rcx.universal_region_relations.outlives(r1, r2)
                || rcx.universal_region_relations.outlives(r2, r1)
        })
    });
    // Now we're left with `['static, 'c]`. Pick `'c` as the minimum!
    let Some(min_choice) = totally_ordered_subset.reduce(|r1, r2| {
        let r1_outlives_r2 = rcx.universal_region_relations.outlives(r1, r2);
        let r2_outlives_r1 = rcx.universal_region_relations.outlives(r2, r1);
        match (r1_outlives_r2, r2_outlives_r1) {
            (true, true) => r1.min(r2),
            (true, false) => r2,
            (false, true) => r1,
            (false, false) => bug!("incomparable regions in total order"),
        }
    }) else {
        debug!("no unique minimum choice");
        return;
    };

    debug!(?min_choice);
    // Lift the member region to be at least as large as this `min_choice` by directly
    // mutating the `scc_values` as we compute it. This acts as if we've added a
    // `'member: 'min_choice` while not recomputing sccs. This means different sccs
    // may now actually be equal.
    let min_choice_scc = rcx.constraint_sccs.scc(min_choice);
    rcx.scc_values.add_region(member, min_choice_scc);
}

struct CollectMemberConstraintsVisitor<'a, 'b, 'tcx> {
    rcx: &'a RegionCtxt<'a, 'tcx>,
    defining_use: &'b DefiningUse<'tcx>,
    member_constraints: &'a mut FxHashMap<ConstraintSccIndex, Vec<&'b DefiningUse<'tcx>>>,
}
impl<'tcx> CollectMemberConstraintsVisitor<'_, '_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.rcx.infcx.tcx
    }
    fn visit_closure_args(&mut self, def_id: DefId, args: GenericArgsRef<'tcx>) {
        let generics = self.cx().generics_of(def_id);
        for arg in args.iter().skip(generics.parent_count) {
            arg.visit_with(self);
        }
    }
}
impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for CollectMemberConstraintsVisitor<'_, '_, 'tcx> {
    fn visit_region(&mut self, r: Region<'tcx>) {
        match r.kind() {
            ty::ReBound(..) => return,
            ty::ReVar(vid) => {
                let scc = self.rcx.constraint_sccs.scc(vid);
                self.member_constraints.entry(scc).or_default().push(self.defining_use);
            }
            _ => unreachable!(),
        }
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return;
        }

        match *ty.kind() {
            ty::Closure(def_id, args)
            | ty::CoroutineClosure(def_id, args)
            | ty::Coroutine(def_id, args) => self.visit_closure_args(def_id, args),

            ty::Alias(kind, ty::AliasTy { def_id, args, .. })
                if let Some(variances) = self.cx().opt_alias_variances(kind, def_id) =>
            {
                // Skip lifetime parameters that are not captured, since they do
                // not need member constraints registered for them; we'll erase
                // them (and hopefully in the future replace them with placeholders).
                for (&v, arg) in std::iter::zip(variances, args.iter()) {
                    if v != ty::Bivariant {
                        arg.visit_with(self)
                    }
                }
            }

            _ => ty.super_visit_with(self),
        }
    }
}
