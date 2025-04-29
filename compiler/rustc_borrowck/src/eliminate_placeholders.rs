//! Logic for lowering higher-kinded outlives constraints
//! (with placeholders and universes) and turn them into regular
//! outlives constraints.
//!
//! This logic is provisional and should be removed once the trait
//! solver can handle this kind of constraint.
use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::graph::scc;
use rustc_data_structures::graph::scc::Sccs;
use rustc_index::IndexVec;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{RegionVid, UniverseIndex};
use tracing::debug;

use crate::constraints::{ConstraintSccIndex, OutlivesConstraintSet};
use crate::consumers::OutlivesConstraint;
use crate::diagnostics::UniverseInfo;
use crate::member_constraints::MemberConstraintSet;
use crate::region_infer::values::{LivenessValues, PlaceholderIndices};
use crate::region_infer::{ConstraintSccs, RegionDefinition, TypeTest};
use crate::ty::VarianceDiagInfo;
use crate::type_check::free_region_relations::UniversalRegionRelations;
use crate::type_check::{Locations, MirTypeckRegionConstraints};
use crate::universal_regions::UniversalRegions;
use crate::{BorrowckInferCtxt, NllRegionVariableOrigin};

/// A set of outlives constraints after rewriting to remove
/// higher-kinded constraints.
pub(crate) struct LoweredConstraints<'tcx> {
    pub(crate) constraint_sccs: Sccs<RegionVid, ConstraintSccIndex>,
    pub(crate) definitions: Frozen<IndexVec<RegionVid, RegionDefinition<'tcx>>>,
    pub(crate) scc_annotations: IndexVec<ConstraintSccIndex, RegionTracker>,
    pub(crate) member_constraints: MemberConstraintSet<'tcx, RegionVid>,
    pub(crate) outlives_constraints: OutlivesConstraintSet<'tcx>,
    pub(crate) type_tests: Vec<TypeTest<'tcx>>,
    pub(crate) liveness_constraints: LivenessValues,
    pub(crate) universe_causes: FxIndexMap<UniverseIndex, UniverseInfo<'tcx>>,
    pub(crate) placeholder_indices: PlaceholderIndices,
}

impl<'d, 'tcx, A: scc::Annotation> SccAnnotations<'d, 'tcx, A> {
    pub(crate) fn init(definitions: &'d IndexVec<RegionVid, RegionDefinition<'tcx>>) -> Self {
        Self { scc_to_annotation: IndexVec::new(), definitions }
    }
}

/// A Visitor for SCC annotation construction.
pub(crate) struct SccAnnotations<'d, 'tcx, A: scc::Annotation> {
    pub(crate) scc_to_annotation: IndexVec<ConstraintSccIndex, A>,
    definitions: &'d IndexVec<RegionVid, RegionDefinition<'tcx>>,
}

impl scc::Annotations<RegionVid> for SccAnnotations<'_, '_, RegionTracker> {
    fn new(&self, element: RegionVid) -> RegionTracker {
        RegionTracker::new(element, &self.definitions[element])
    }

    fn annotate_scc(&mut self, scc: ConstraintSccIndex, annotation: RegionTracker) {
        let idx = self.scc_to_annotation.push(annotation);
        assert!(idx == scc);
    }

    type Ann = RegionTracker;
    type SccIdx = ConstraintSccIndex;
}

/// An annotation for region graph SCCs that tracks
/// the values of its elements. This annotates a single SCC.
#[derive(Copy, Debug, Clone)]
pub(crate) struct RegionTracker {
    /// The largest universe of a placeholder reached from this SCC.
    /// This includes placeholders within this SCC.
    max_placeholder_universe_reached: UniverseIndex,

    /// The smallest universe index reachable form the nodes of this SCC.
    min_reachable_universe: UniverseIndex,

    /// The representative Region Variable Id for this SCC. We prefer
    /// placeholders over existentially quantified variables, otherwise
    ///  it's the one with the smallest Region Variable ID.
    pub(crate) representative: RegionVid,

    /// Is the current representative a placeholder?
    representative_is_placeholder: bool,

    /// Is the current representative existentially quantified?
    representative_is_existential: bool,
}

impl RegionTracker {
    pub(crate) fn new(rvid: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        let (representative_is_placeholder, representative_is_existential) = match definition.origin
        {
            NllRegionVariableOrigin::FreeRegion => (false, false),
            NllRegionVariableOrigin::Placeholder(_) => (true, false),
            NllRegionVariableOrigin::Existential { .. } => (false, true),
        };

        let placeholder_universe =
            if representative_is_placeholder { definition.universe } else { UniverseIndex::ROOT };

        Self {
            max_placeholder_universe_reached: placeholder_universe,
            min_reachable_universe: definition.universe,
            representative: rvid,
            representative_is_placeholder,
            representative_is_existential,
        }
    }

    /// The smallest-indexed universe reachable from and/or in this SCC.
    pub(crate) fn min_universe(self) -> UniverseIndex {
        self.min_reachable_universe
    }

    fn merge_min_max_seen(&mut self, other: &Self) {
        self.max_placeholder_universe_reached = std::cmp::max(
            self.max_placeholder_universe_reached,
            other.max_placeholder_universe_reached,
        );

        self.min_reachable_universe =
            std::cmp::min(self.min_reachable_universe, other.min_reachable_universe);
    }

    /// Returns `true` if during the annotated SCC reaches a placeholder
    /// with a universe larger than the smallest reachable one, `false` otherwise.
    pub(crate) fn has_incompatible_universes(&self) -> bool {
        self.min_universe().cannot_name(self.max_placeholder_universe_reached)
    }

    /// Determine if the tracked universes of the two SCCs
    /// are compatible.
    pub(crate) fn universe_compatible_with(&self, other: Self) -> bool {
        self.min_universe().can_name(other.min_universe())
            || self.min_universe().can_name(other.max_placeholder_universe_reached)
    }
}

impl scc::Annotation for RegionTracker {
    fn merge_scc(mut self, mut other: Self) -> Self {
        // Prefer any placeholder over any existential
        if other.representative_is_placeholder && self.representative_is_existential {
            other.merge_min_max_seen(&self);
            return other;
        }

        if self.representative_is_placeholder && other.representative_is_existential
            || (self.representative <= other.representative)
        {
            self.merge_min_max_seen(&other);
            return self;
        }
        other.merge_min_max_seen(&self);
        other
    }

    fn merge_reached(mut self, other: Self) -> Self {
        // No update to in-component values, only add seen values.
        self.merge_min_max_seen(&other);
        self
    }
}

/// Determines if the region variable definitions contain
/// placeholers, and compute them for later use.
fn region_definitions<'tcx>(
    universal_regions: &UniversalRegions<'tcx>,
    infcx: &BorrowckInferCtxt<'tcx>,
) -> (Frozen<IndexVec<RegionVid, RegionDefinition<'tcx>>>, bool) {
    let var_infos = infcx.get_region_var_infos();
    // Create a RegionDefinition for each inference variable. This happens here because
    // it allows us to sneak in a cheap check for placeholders. Otherwise, its proper home
    // is in `RegionInferenceContext::new()`, probably.
    let mut definitions = IndexVec::with_capacity(var_infos.len());
    let mut has_placeholders = false;

    for info in var_infos.iter() {
        let definition = RegionDefinition::new(info);
        has_placeholders |= matches!(definition.origin, NllRegionVariableOrigin::Placeholder(_));
        definitions.push(definition);
    }

    // Add external names from universal regions in fun function definitions.
    for (external_name, variable) in universal_regions.named_universal_regions_iter() {
        debug!("region {:?} has external name {:?}", variable, external_name);
        definitions[variable].external_name = Some(external_name);
    }
    (Frozen::freeze(definitions), has_placeholders)
}

/// This method handles Universe errors by rewriting the constraint
/// graph. For each strongly connected component in the constraint
/// graph such that there is a series of constraints
///    A: B: C: ... : X  where
/// A's universe is smaller than X's and A is a placeholder,
/// add a constraint that A: 'static. This is a safe upper bound
/// in the face of borrow checker/trait solver limitations that will
/// eventually go away.
///
/// For a more precise definition, see the documentation for
/// [`RegionTracker`] and its methods!.
///
/// Since universes can also be involved in errors (if one placeholder
/// transitively outlives another), this function also flags those.
///
/// Additionally, it similarly rewrites type-tests.
///
/// This edge case used to be handled during constraint propagation
/// by iterating over the strongly connected components in the constraint
/// graph while maintaining a set of bookkeeping mappings similar
/// to what is stored in `RegionTracker` and manually adding 'sttaic as
/// needed.
///
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
pub(crate) fn rewrite_higher_kinded_outlives_as_constraints<'tcx>(
    constraints: MirTypeckRegionConstraints<'tcx>,
    universal_region_relations: &Frozen<UniversalRegionRelations<'tcx>>,
    infcx: &BorrowckInferCtxt<'tcx>,
) -> LoweredConstraints<'tcx> {
    let universal_regions = &universal_region_relations.universal_regions;
    let (definitions, has_placeholders) = region_definitions(universal_regions, infcx);

    let MirTypeckRegionConstraints {
        placeholder_indices,
        placeholder_index_to_region: _,
        liveness_constraints,
        mut outlives_constraints,
        mut member_constraints,
        universe_causes,
        type_tests,
    } = constraints;

    if let Some(guar) = universal_regions.tainted_by_errors() {
        debug!("Universal regions tainted by errors; removing constraints!");
        // Suppress unhelpful extra errors in `infer_opaque_types` by clearing out all
        // outlives bounds that we may end up checking.
        outlives_constraints = Default::default();
        member_constraints = Default::default();

        // Also taint the entire scope.
        infcx.set_tainted_by_errors(guar);
    }

    let fr_static = universal_regions.fr_static;
    let compute_sccs =
        |constraints: &OutlivesConstraintSet<'tcx>,
         annotations: &mut SccAnnotations<'_, 'tcx, RegionTracker>| {
            ConstraintSccs::new_with_annotation(
                &constraints.graph(definitions.len()).region_graph(constraints, fr_static),
                annotations,
            )
        };

    // This code structure is a bit convoluted because it allows for a planned
    // future change where the early return here has a different type of annotation
    // that does much less work.
    if !has_placeholders {
        debug!("No placeholder regions found; skipping rewriting logic!");
        let mut scc_annotations = SccAnnotations::init(&definitions);
        let constraint_sccs = compute_sccs(&outlives_constraints, &mut scc_annotations);

        return LoweredConstraints {
            type_tests,
            member_constraints,
            constraint_sccs,
            scc_annotations: scc_annotations.scc_to_annotation,
            definitions,
            outlives_constraints,
            liveness_constraints,
            universe_causes,
            placeholder_indices,
        };
    }
    debug!("Placeholders present; activating placeholder handling logic!");

    let mut annotations = SccAnnotations::init(&definitions);
    let sccs = compute_sccs(&outlives_constraints, &mut annotations);

    let outlives_static =
        rewrite_outlives(&sccs, &annotations, fr_static, &mut outlives_constraints);

    let (sccs, scc_annotations) = if !outlives_static.is_empty() {
        debug!("The following SCCs had :'static constraints added: {:?}", outlives_static);
        let mut annotations = SccAnnotations::init(&definitions);

        // We changed the constraint set and so must recompute SCCs.
        // Optimisation opportunity: if we can add them incrementally (and that's
        // possible because edges to 'static always only merge SCCs into 'static),
        // we would potentially save a lot of work here.
        (compute_sccs(&outlives_constraints, &mut annotations), annotations.scc_to_annotation)
    } else {
        // If we didn't add any back-edges; no more work needs doing
        debug!("No constraints rewritten!");
        (sccs, annotations.scc_to_annotation)
    };

    LoweredConstraints {
        constraint_sccs: sccs,
        definitions,
        scc_annotations,
        member_constraints,
        outlives_constraints,
        type_tests,
        liveness_constraints,
        universe_causes,
        placeholder_indices,
    }
}

fn rewrite_outlives<'tcx>(
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    annotations: &SccAnnotations<'_, '_, RegionTracker>,
    fr_static: RegionVid,
    outlives_constraints: &mut OutlivesConstraintSet<'tcx>,
) -> FxHashSet<ConstraintSccIndex> {
    // Changed to `true` if we added any constraints to `self` and need to
    // recompute SCCs.
    let mut outlives_static = FxHashSet::default();

    let annotations = &annotations.scc_to_annotation;

    for scc in sccs.all_sccs() {
        // No point in adding 'static: 'static!
        // This micro-optimisation makes somewhat sense
        // because static outlives *everything*.
        if scc == sccs.scc(fr_static) {
            continue;
        }

        let annotation = annotations[scc];

        // If this SCC participates in a universe violation,
        // e.g. if it reaches a region with a universe smaller than
        // the largest region reached, add a requirement that it must
        // outlive `'static`.
        if annotation.has_incompatible_universes() {
            // Optimisation opportunity: this will add more constraints than
            // needed for correctness, since an SCC upstream of another with
            // a universe violation will "infect" its downstream SCCs to also
            // outlive static.
            outlives_static.insert(scc);
            let scc_representative_outlives_static = OutlivesConstraint {
                sup: annotation.representative,
                sub: fr_static,
                category: ConstraintCategory::IllegalUniverse,
                locations: Locations::All(rustc_span::DUMMY_SP),
                span: rustc_span::DUMMY_SP,
                variance_info: VarianceDiagInfo::None,
                from_closure: false,
            };
            outlives_constraints.push(scc_representative_outlives_static);
        }
    }
    outlives_static
}
