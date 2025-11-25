//! Logic for lowering higher-kinded outlives constraints
//! (with placeholders and universes) and turn them into regular
//! outlives constraints.
use std::cmp;

use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::graph::scc;
use rustc_data_structures::graph::scc::Sccs;
use rustc_index::IndexVec;
use rustc_infer::infer::RegionVariableOrigin;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{RegionVid, UniverseIndex};
use tracing::{debug, trace};

use crate::constraints::{ConstraintSccIndex, OutlivesConstraintSet};
use crate::consumers::OutlivesConstraint;
use crate::diagnostics::{RegionErrorKind, RegionErrors, UniverseInfo};
use crate::region_infer::values::LivenessValues;
use crate::region_infer::{ConstraintSccs, RegionDefinition, Representative, TypeTest};
use crate::ty::VarianceDiagInfo;
use crate::type_check::free_region_relations::UniversalRegionRelations;
use crate::type_check::{Locations, MirTypeckRegionConstraints};
use crate::universal_regions::UniversalRegions;
use crate::{BorrowckInferCtxt, NllRegionVariableOrigin};

pub(crate) trait RegionSccs {
    fn representative(&self, scc: ConstraintSccIndex) -> RegionVid;
    fn reachable_placeholders(&self, _scc: ConstraintSccIndex) -> Option<PlaceholderReachability> {
        None
    }
    /// The largest universe this SCC can name. It's the smallest
    /// largest nameable universe of any reachable region, or
    /// `max_nameable(r) = min (max_nameable(r') for r' reachable from r)`
    fn max_nameable_universe(&self, _scc: ConstraintSccIndex) -> UniverseIndex {
        UniverseIndex::ROOT
    }
    fn max_placeholder_universe_reached(&self, _scc: ConstraintSccIndex) -> UniverseIndex {
        UniverseIndex::ROOT
    }
}

impl RegionSccs for IndexVec<ConstraintSccIndex, RegionTracker> {
    fn representative(&self, scc: ConstraintSccIndex) -> RegionVid {
        self[scc].representative.rvid()
    }

    fn reachable_placeholders(&self, scc: ConstraintSccIndex) -> Option<PlaceholderReachability> {
        self[scc].reachable_placeholders
    }

    fn max_nameable_universe(&self, scc: ConstraintSccIndex) -> UniverseIndex {
        // Note that this is stricter than it might need to be!
        self[scc].min_max_nameable_universe
    }

    fn max_placeholder_universe_reached(&self, scc: ConstraintSccIndex) -> UniverseIndex {
        self[scc].reachable_placeholders.map(|p| p.max_universe.u).unwrap_or(UniverseIndex::ROOT)
    }
}

impl RegionSccs for IndexVec<ConstraintSccIndex, Representative> {
    fn representative(&self, scc: ConstraintSccIndex) -> RegionVid {
        self[scc].rvid()
    }
}

impl FromRegionDefinition for Representative {
    fn new(rvid: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        Representative::new(rvid, definition)
    }
}

/// A set of outlives constraints after rewriting to remove
/// higher-kinded constraints.
pub(crate) struct LoweredConstraints<'tcx> {
    pub(crate) constraint_sccs: Sccs<RegionVid, ConstraintSccIndex>,
    pub(crate) definitions: Frozen<IndexVec<RegionVid, RegionDefinition<'tcx>>>,
    pub(crate) scc_annotations: Box<dyn RegionSccs>,
    pub(crate) outlives_constraints: Frozen<OutlivesConstraintSet<'tcx>>,
    pub(crate) type_tests: Vec<TypeTest<'tcx>>,
    pub(crate) liveness_constraints: LivenessValues,
    pub(crate) universe_causes: FxIndexMap<UniverseIndex, UniverseInfo<'tcx>>,
}

impl<'d, 'tcx, A: scc::Annotation> SccAnnotations<'d, 'tcx, A> {
    pub(crate) fn init(definitions: &'d IndexVec<RegionVid, RegionDefinition<'tcx>>) -> Self {
        Self { scc_to_annotation: IndexVec::new(), definitions }
    }
}

trait FromRegionDefinition {
    fn new(rvid: RegionVid, definition: &RegionDefinition<'_>) -> Self;
}

/// A Visitor for SCC annotation construction.
pub(crate) struct SccAnnotations<'d, 'tcx, A: scc::Annotation> {
    pub(crate) scc_to_annotation: IndexVec<ConstraintSccIndex, A>,
    definitions: &'d IndexVec<RegionVid, RegionDefinition<'tcx>>,
}
impl<A: scc::Annotation + FromRegionDefinition> scc::Annotations<RegionVid>
    for SccAnnotations<'_, '_, A>
{
    fn new(&self, element: RegionVid) -> A {
        A::new(element, &self.definitions[element])
    }

    fn annotate_scc(&mut self, scc: ConstraintSccIndex, annotation: A) {
        let idx = self.scc_to_annotation.push(annotation);
        assert!(idx == scc);
    }

    type Ann = A;
    type SccIdx = ConstraintSccIndex;
}

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlaceholderReachability {
    /// The largest-universed placeholder we can reach
    max_universe: RegionWithUniverse,

    /// The placeholder with the smallest ID
    pub(crate) min_placeholder: RegionVid,

    /// The placeholder with the largest ID
    pub(crate) max_placeholder: RegionVid,
}

impl PlaceholderReachability {
    /// Merge the reachable placeholders of two graph components.
    fn merge(&mut self, other: &PlaceholderReachability) {
        self.max_universe = self.max_universe.max(other.max_universe);
        self.min_placeholder = self.min_placeholder.min(other.min_placeholder);
        self.max_placeholder = self.max_placeholder.max(other.max_placeholder);
    }
}

/// A region with its universe, ordered fist by largest unverse, then
/// by smallest region (reverse region id order).
#[derive(Copy, Debug, Clone, PartialEq, Eq)]
struct RegionWithUniverse {
    u: UniverseIndex,
    r: RegionVid,
}

impl Ord for RegionWithUniverse {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if self.u.cmp(&other.u) == cmp::Ordering::Equal {
            self.r.cmp(&other.r).reverse()
        } else {
            self.u.cmp(&other.u)
        }
    }
}

impl PartialOrd for RegionWithUniverse {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// An annotation for region graph SCCs that tracks
/// the values of its elements. This annotates a single SCC.
#[derive(Copy, Debug, Clone)]
pub(crate) struct RegionTracker {
    pub(crate) reachable_placeholders: Option<PlaceholderReachability>,

    /// The smallest max nameable universe of all
    /// regions reachable from this SCC.
    min_max_nameable_universe: UniverseIndex,

    /// The worst-nameable (highest univers'd) placeholder region in this SCC.
    is_placeholder: Option<RegionWithUniverse>,

    /// The worst-naming (min univers'd) existential region we reach.
    worst_existential: Option<(UniverseIndex, RegionVid)>,

    /// The representative Region Variable Id for this SCC.
    pub(crate) representative: Representative,
}

impl FromRegionDefinition for RegionTracker {
    fn new(rvid: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        use NllRegionVariableOrigin::*;

        let min_max_nameable_universe = definition.universe;
        let representative = Representative::new(rvid, definition);
        let universe_and_rvid = RegionWithUniverse { r: rvid, u: definition.universe };

        match definition.origin {
            FreeRegion => Self {
                reachable_placeholders: None,
                min_max_nameable_universe,
                is_placeholder: None,
                worst_existential: None,
                representative,
            },
            Placeholder(_) => Self {
                reachable_placeholders: Some(PlaceholderReachability {
                    max_universe: universe_and_rvid,
                    min_placeholder: rvid,
                    max_placeholder: rvid,
                }),
                min_max_nameable_universe,
                is_placeholder: Some(universe_and_rvid),
                worst_existential: None,
                representative,
            },
            Existential { .. } => Self {
                reachable_placeholders: None,
                min_max_nameable_universe,
                is_placeholder: None,
                worst_existential: Some((definition.universe, rvid)),
                representative,
            },
        }
    }
}

impl scc::Annotation for RegionTracker {
    fn update_scc(&mut self, other: &Self) {
        trace!("{:?} << {:?}", self.representative, other.representative);
        self.representative.update_scc(&other.representative);
        self.is_placeholder = self.is_placeholder.max(other.is_placeholder);
        self.update_reachable(other); // SCC membership implies reachability.
    }

    #[inline(always)]
    fn update_reachable(&mut self, other: &Self) {
        self.worst_existential = self
            .worst_existential
            .xor(other.worst_existential)
            .or_else(|| self.worst_existential.min(other.worst_existential));
        self.min_max_nameable_universe =
            self.min_max_nameable_universe.min(other.min_max_nameable_universe);
        match (self.reachable_placeholders.as_mut(), other.reachable_placeholders.as_ref()) {
            (None, None) | (Some(_), None) => (),
            (None, Some(theirs)) => self.reachable_placeholders = Some(*theirs),
            (Some(ours), Some(theirs)) => ours.merge(theirs),
        };
    }
}

/// Determines if the region variable definitions contain
/// placeholders, and compute them for later use.
// FIXME: This is also used by opaque type handling. Move it to a separate file.
pub(super) fn region_definitions<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    universal_regions: &UniversalRegions<'tcx>,
) -> (Frozen<IndexVec<RegionVid, RegionDefinition<'tcx>>>, bool) {
    let var_infos = infcx.get_region_var_infos();
    // Create a RegionDefinition for each inference variable. This happens here because
    // it allows us to sneak in a cheap check for placeholders. Otherwise, its proper home
    // is in `RegionInferenceContext::new()`, probably.
    let mut definitions = IndexVec::with_capacity(var_infos.len());
    let mut has_placeholders = false;

    for info in var_infos.iter() {
        let origin = match info.origin {
            RegionVariableOrigin::Nll(origin) => origin,
            _ => NllRegionVariableOrigin::Existential { name: None },
        };

        let definition = RegionDefinition { origin, universe: info.universe, external_name: None };

        has_placeholders |= matches!(origin, NllRegionVariableOrigin::Placeholder(_));
        definitions.push(definition);
    }

    // Add external names from universal regions in fun function definitions.
    // FIXME: this two-step method is annoying, but I don't know how to avoid it.
    for (external_name, variable) in universal_regions.named_universal_regions_iter() {
        trace!("region {:?} has external name {:?}", variable, external_name);
        definitions[variable].external_name = Some(external_name);
    }
    (Frozen::freeze(definitions), has_placeholders)
}

/// This method handles placeholders by rewriting the constraint
/// graph. For each strongly connected component in the constraint
/// graph such that there is a series of constraints
///    A: B: C: ... : X  where
/// A contains a placeholder whose universe cannot be named by X,
/// add a constraint that A: 'static. This is a safe upper bound
/// in the face of borrow checker/trait solver limitations that will
/// eventually go away.
///
/// For a more precise definition, see the documentation for
/// [`RegionTracker`] and its methods!
///
/// This edge case used to be handled during constraint propagation.
/// It was rewritten as part of the Polonius project with the goal of moving
/// higher-kindedness concerns out of the path of the borrow checker,
/// for two reasons:
///
/// 1. Implementing Polonius is difficult enough without also
///     handling them.
/// 2. The long-term goal is to handle higher-kinded concerns
///     in the trait solver, where they belong. This avoids
///     logic duplication and allows future trait solvers
///     to compute better bounds than for example our
///     "must outlive 'static" here.
///
/// This code is a stop-gap measure in preparation for the future trait solver.
///
/// Every constraint added by this method is an internal `IllegalUniverse` constraint.
pub(crate) fn compute_sccs_applying_placeholder_outlives_constraints<'tcx>(
    constraints: MirTypeckRegionConstraints<'tcx>,
    universal_region_relations: &Frozen<UniversalRegionRelations<'tcx>>,
    infcx: &BorrowckInferCtxt<'tcx>,
    errors_buffer: &mut RegionErrors<'tcx>,
) -> LoweredConstraints<'tcx> {
    let universal_regions = &universal_region_relations.universal_regions;
    let (definitions, has_placeholders) = region_definitions(infcx, universal_regions);

    let MirTypeckRegionConstraints {
        liveness_constraints,
        mut outlives_constraints,
        universe_causes,
        type_tests,
        placeholder_to_region: _,
    } = constraints;

    let fr_static = universal_regions.fr_static;
    let compute_sccs =
        |constraints: &OutlivesConstraintSet<'tcx>,
         annotations: &mut SccAnnotations<'_, 'tcx, RegionTracker>| {
            ConstraintSccs::new_with_annotation(
                &constraints.graph(definitions.len()).region_graph(constraints, fr_static),
                annotations,
            )
        };

    if !has_placeholders {
        debug!("No placeholder regions found; skipping rewriting logic!");

        // We skip the extra logic and only record representatives.
        let mut scc_annotations: SccAnnotations<'_, '_, Representative> =
            SccAnnotations::init(&definitions);
        let constraint_sccs = ConstraintSccs::new_with_annotation(
            &outlives_constraints
                .graph(definitions.len())
                .region_graph(&outlives_constraints, fr_static),
            &mut scc_annotations,
        );

        return LoweredConstraints {
            type_tests,
            constraint_sccs,
            scc_annotations: Box::new(scc_annotations.scc_to_annotation),
            definitions,
            outlives_constraints: Frozen::freeze(outlives_constraints),
            liveness_constraints,
            universe_causes,
        };
    }
    debug!("Placeholders present; activating placeholder handling logic!");
    let mut scc_annotations = SccAnnotations::init(&definitions);
    let constraint_sccs = compute_sccs(&outlives_constraints, &mut scc_annotations);

    let added_constraints = rewrite_placeholder_outlives(
        &constraint_sccs,
        &scc_annotations,
        fr_static,
        &mut outlives_constraints,
        errors_buffer,
        &definitions,
    );

    let (constraint_sccs, scc_annotations) = if added_constraints {
        let mut annotations = SccAnnotations::init(&definitions);

        // We changed the constraint set and so must recompute SCCs.
        // Optimisation opportunity: if we can add them incrementally (and that's
        // possible because edges to 'static always only merge SCCs into 'static),
        // we would potentially save a lot of work here.
        (compute_sccs(&outlives_constraints, &mut annotations), annotations.scc_to_annotation)
    } else {
        // If we didn't add any back-edges; no more work needs doing
        debug!("No constraints rewritten!");
        (constraint_sccs, scc_annotations.scc_to_annotation)
    };

    LoweredConstraints {
        constraint_sccs,
        definitions,
        scc_annotations: Box::new(scc_annotations),
        outlives_constraints: Frozen::freeze(outlives_constraints),
        type_tests,
        liveness_constraints,
        universe_causes,
    }
}

pub(crate) fn rewrite_placeholder_outlives<'tcx>(
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    annotations: &SccAnnotations<'_, '_, RegionTracker>,
    fr_static: RegionVid,
    outlives_constraints: &mut OutlivesConstraintSet<'tcx>,
    errors_buffer: &mut RegionErrors<'tcx>,
    definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
) -> bool {
    // Changed to `true` if we added any constraints and need to
    // recompute SCCs.
    let mut added_constraints = false;

    let annotations = &annotations.scc_to_annotation;

    for scc in sccs.all_sccs() {
        // No point in adding 'static: 'static!
        // This micro-optimisation makes somewhat sense
        // because static outlives *everything*.
        if scc == sccs.scc(fr_static) {
            continue;
        }

        let annotation = annotations[scc];

        let Some(PlaceholderReachability { min_placeholder, max_placeholder, max_universe }) =
            annotation.reachable_placeholders
        else {
            trace!("No placeholders reached from {scc:?}");
            continue;
        };

        if let Some(us) = annotation.is_placeholder
            && min_placeholder != max_placeholder
        {
            let illegally_outlived_r =
                if min_placeholder == us.r { max_placeholder } else { min_placeholder };
            debug!("Placeholder {us:?} outlives placeholder {illegally_outlived_r:?}");
            errors_buffer.push(RegionErrorKind::PlaceholderOutlivesIllegalRegion {
                longer_fr: us.r,
                illegally_outlived_r,
            });
            // FIXME: investigate if it's an actual improvement to drop early here
            // and stop reporting errors for this SCC since we are guaranteed to
            // have at least one.
        }

        if annotation.min_max_nameable_universe.can_name(max_universe.u) {
            trace!("All placeholders nameable from {scc:?}!");
            continue;
        }

        debug!(
            "Placeholder {max_universe:?} unnameable from {scc:?} represented by {:?}",
            annotation.representative
        );

        // Figure out if we had our universe lowered by an existential
        // that cannot name us, a placeholder. This is an error.
        if let Some((ex_u, ex_r)) = annotation.worst_existential
            && let Some(pl) = annotation.is_placeholder
            && ex_u.cannot_name(pl.u)
        {
            debug!("{pl:?} outlives existential {ex_r:?} that cannot name it!");
            // Prefer the representative region if it's also unnameable.
            let longer_fr = if let Representative::Placeholder(p) = annotation.representative
                && ex_u.cannot_name(definitions[p].universe)
            {
                p
            } else {
                pl.r
            };
            errors_buffer.push(RegionErrorKind::PlaceholderOutlivesIllegalRegion {
                longer_fr,
                illegally_outlived_r: ex_r,
            });
            // FIXME: we could `continue` here since there is no point in adding
            // 'r: 'static for this SCC now that it's already outlived an
            // existential it shouldn't, but we do anyway for compatibility with
            // earlier versions' output.
        };

        let representative_rvid = annotation.representative.rvid();

        if representative_rvid == max_universe.r {
            assert!(matches!(annotation.representative, Representative::Placeholder(_)));
            // The unnameable placeholder *is* the representative.
            // If this SCC is represented by a placeholder `p` which cannot be named,
            // from its own SCC, `p` must have at some point reached a/an:
            // - existential region that could not name it (the `if` above)
            // - free region that lowered its universe (will flag an error in region
            //   inference since `p` isn't empty)
            // - another placeholder (will flag an error above, but will reach here).
            //
            // To avoid adding the invalid constraint "`'p: 'static` due to `'p` being
            // unnameable from the SCC represented by `'p`", we nope out early here
            // at no risk of soundness issues since at this point all paths lead
            // to an error.
            continue;
        }
        // This SCC outlives a placeholder it can't name and must outlive 'static.

        // FIXME: if we can extract a useful blame span here, future error
        // reporting and constraint search can be simplified.

        added_constraints = true;
        outlives_constraints.push(OutlivesConstraint {
            sup: representative_rvid,
            sub: fr_static,
            category: ConstraintCategory::OutlivesUnnameablePlaceholder(max_universe.r),
            locations: Locations::All(rustc_span::DUMMY_SP),
            span: rustc_span::DUMMY_SP,
            variance_info: VarianceDiagInfo::None,
            from_closure: false,
        });
    }
    added_constraints
}
