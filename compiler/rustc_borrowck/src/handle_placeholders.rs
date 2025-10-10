//! Logic for lowering higher-kinded outlives constraints
//! (with placeholders and universes) and turn them into regular
//! outlives constraints.
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
use crate::diagnostics::UniverseInfo;
use crate::region_infer::values::{LivenessValues, PlaceholderIndices};
use crate::region_infer::{ConstraintSccs, RegionDefinition, Representative, TypeTest};
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
    pub(crate) outlives_constraints: Frozen<OutlivesConstraintSet<'tcx>>,
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

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
enum PlaceholderReachability {
    /// This SCC reaches no placeholders.
    NoPlaceholders,
    /// This SCC reaches at least one placeholder.
    Placeholders {
        /// The largest-universed placeholder we can reach
        max_universe: (UniverseIndex, RegionVid),

        /// The placeholder with the smallest ID
        min_placeholder: RegionVid,

        /// The placeholder with the largest ID
        max_placeholder: RegionVid,
    },
}

impl PlaceholderReachability {
    /// Merge the reachable placeholders of two graph components.
    fn merge(self, other: PlaceholderReachability) -> PlaceholderReachability {
        use PlaceholderReachability::*;
        match (self, other) {
            (NoPlaceholders, NoPlaceholders) => NoPlaceholders,
            (NoPlaceholders, p @ Placeholders { .. })
            | (p @ Placeholders { .. }, NoPlaceholders) => p,
            (
                Placeholders {
                    min_placeholder: min_pl,
                    max_placeholder: max_pl,
                    max_universe: max_u,
                },
                Placeholders { min_placeholder, max_placeholder, max_universe },
            ) => Placeholders {
                min_placeholder: min_pl.min(min_placeholder),
                max_placeholder: max_pl.max(max_placeholder),
                max_universe: max_u.max(max_universe),
            },
        }
    }

    fn max_universe(&self) -> Option<(UniverseIndex, RegionVid)> {
        match self {
            Self::NoPlaceholders => None,
            Self::Placeholders { max_universe, .. } => Some(*max_universe),
        }
    }

    /// If we have reached placeholders, determine if they can
    /// be named from this universe.
    fn can_be_named_by(&self, from: UniverseIndex) -> bool {
        self.max_universe()
            .is_none_or(|(max_placeholder_universe, _)| from.can_name(max_placeholder_universe))
    }
}

/// An annotation for region graph SCCs that tracks
/// the values of its elements. This annotates a single SCC.
#[derive(Copy, Debug, Clone)]
pub(crate) struct RegionTracker {
    reachable_placeholders: PlaceholderReachability,

    /// The largest universe nameable from this SCC.
    /// It is the smallest nameable universes of all
    /// existential regions reachable from it. Small Rvids are preferred.
    max_nameable_universe: (UniverseIndex, RegionVid),

    /// The representative Region Variable Id for this SCC.
    pub(crate) representative: Representative,
}

impl RegionTracker {
    pub(crate) fn new(rvid: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        let reachable_placeholders =
            if matches!(definition.origin, NllRegionVariableOrigin::Placeholder(_)) {
                PlaceholderReachability::Placeholders {
                    max_universe: (definition.universe, rvid),
                    min_placeholder: rvid,
                    max_placeholder: rvid,
                }
            } else {
                PlaceholderReachability::NoPlaceholders
            };

        Self {
            reachable_placeholders,
            max_nameable_universe: (definition.universe, rvid),
            representative: Representative::new(rvid, definition),
        }
    }

    /// The largest universe this SCC can name. It's the smallest
    /// largest nameable universe of any reachable region, or
    /// `max_nameable(r) = min (max_nameable(r') for r' reachable from r)`
    pub(crate) fn max_nameable_universe(self) -> UniverseIndex {
        self.max_nameable_universe.0
    }

    pub(crate) fn max_placeholder_universe_reached(self) -> UniverseIndex {
        if let Some((universe, _)) = self.reachable_placeholders.max_universe() {
            universe
        } else {
            UniverseIndex::ROOT
        }
    }

    /// Determine if we can name all the placeholders in `other`.
    pub(crate) fn can_name_all_placeholders(&self, other: Self) -> bool {
        other.reachable_placeholders.can_be_named_by(self.max_nameable_universe.0)
    }

    /// If this SCC reaches a placeholder it can't name, return it.
    fn unnameable_placeholder(&self) -> Option<(UniverseIndex, RegionVid)> {
        self.reachable_placeholders.max_universe().filter(|&(placeholder_universe, _)| {
            !self.max_nameable_universe().can_name(placeholder_universe)
        })
    }
}

impl scc::Annotation for RegionTracker {
    fn merge_scc(self, other: Self) -> Self {
        trace!("{:?} << {:?}", self.representative, other.representative);

        Self {
            representative: self.representative.min(other.representative),
            max_nameable_universe: self.max_nameable_universe.min(other.max_nameable_universe),
            reachable_placeholders: self.reachable_placeholders.merge(other.reachable_placeholders),
        }
    }

    fn merge_reached(self, other: Self) -> Self {
        Self {
            max_nameable_universe: self.max_nameable_universe.min(other.max_nameable_universe),
            reachable_placeholders: self.reachable_placeholders.merge(other.reachable_placeholders),
            representative: self.representative,
        }
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
        debug!("region {:?} has external name {:?}", variable, external_name);
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
) -> LoweredConstraints<'tcx> {
    let universal_regions = &universal_region_relations.universal_regions;
    let (definitions, has_placeholders) = region_definitions(infcx, universal_regions);

    let MirTypeckRegionConstraints {
        placeholder_indices,
        placeholder_index_to_region: _,
        liveness_constraints,
        mut outlives_constraints,
        universe_causes,
        type_tests,
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

    let mut scc_annotations = SccAnnotations::init(&definitions);
    let constraint_sccs = compute_sccs(&outlives_constraints, &mut scc_annotations);

    // This code structure is a bit convoluted because it allows for a planned
    // future change where the early return here has a different type of annotation
    // that does much less work.
    if !has_placeholders {
        debug!("No placeholder regions found; skipping rewriting logic!");

        return LoweredConstraints {
            type_tests,
            constraint_sccs,
            scc_annotations: scc_annotations.scc_to_annotation,
            definitions,
            outlives_constraints: Frozen::freeze(outlives_constraints),
            liveness_constraints,
            universe_causes,
            placeholder_indices,
        };
    }
    debug!("Placeholders present; activating placeholder handling logic!");

    let added_constraints = rewrite_placeholder_outlives(
        &constraint_sccs,
        &scc_annotations,
        fr_static,
        &mut outlives_constraints,
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
        scc_annotations,
        outlives_constraints: Frozen::freeze(outlives_constraints),
        type_tests,
        liveness_constraints,
        universe_causes,
        placeholder_indices,
    }
}

fn rewrite_placeholder_outlives<'tcx>(
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    annotations: &SccAnnotations<'_, '_, RegionTracker>,
    fr_static: RegionVid,
    outlives_constraints: &mut OutlivesConstraintSet<'tcx>,
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

        let Some((max_u, max_u_rvid)) = annotation.unnameable_placeholder() else {
            continue;
        };

        debug!(
            "Placeholder universe {max_u:?} is too large for its SCC, represented by {:?}",
            annotation.representative
        );

        // We only add one `r: 'static` constraint per SCC, where `r` is the SCC representative.
        // That constraint is annotated with some placeholder `unnameable` where
        // `unnameable` is unnameable from `r` and there is a path in the constraint graph
        // between them.
        //
        // There is one exception; if some other region in this SCC can't name `'r`, then
        // we pick the region with the smallest universe in the SCC, so that a path can
        // always start in `'r` to find a motivation that isn't cyclic.
        let blame_to = if annotation.representative.rvid() == max_u_rvid {
            // Assertion: the region that lowered our universe is an existential one and we are a placeholder!

            // The SCC's representative is not nameable from some region
            // that ends up in the SCC.
            let small_universed_rvid = annotation.max_nameable_universe.1;
            debug!(
                "{small_universed_rvid:?} lowered our universe to {:?}",
                annotation.max_nameable_universe()
            );
            small_universed_rvid
        } else {
            // `max_u_rvid` is not nameable by the SCC's representative.
            max_u_rvid
        };

        // FIXME: if we can extract a useful blame span here, future error
        // reporting and constraint search can be simplified.

        added_constraints = true;
        outlives_constraints.push(OutlivesConstraint {
            sup: annotation.representative.rvid(),
            sub: fr_static,
            category: ConstraintCategory::OutlivesUnnameablePlaceholder(blame_to),
            locations: Locations::All(rustc_span::DUMMY_SP),
            span: rustc_span::DUMMY_SP,
            variance_info: VarianceDiagInfo::None,
            from_closure: false,
        });
    }
    added_constraints
}
