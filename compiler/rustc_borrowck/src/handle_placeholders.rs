//! Logic for lowering higher-kinded outlives constraints
//! (with placeholders and universes) and turn them into regular
//! outlives constraints.
use std::cell::OnceCell;
use std::collections::VecDeque;

use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::graph::scc;
use rustc_data_structures::graph::scc::Sccs;
use rustc_index::IndexVec;
use rustc_infer::infer::RegionVariableOrigin;
use rustc_middle::bug;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{RegionVid, UniverseIndex};
use tracing::{debug, instrument, trace};

use crate::constraints::graph::{ConstraintGraph, Normal, RegionGraph};
use crate::constraints::{ConstraintSccIndex, OutlivesConstraintSet};
use crate::consumers::OutlivesConstraint;
use crate::diagnostics::{RegionErrorKind, RegionErrors, UniverseInfo};
use crate::member_constraints::MemberConstraintSet;
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
    pub(crate) member_constraints: MemberConstraintSet<'tcx, RegionVid>,
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
                min_placeholder: core::cmp::min(min_pl, min_placeholder),
                max_placeholder: core::cmp::max(max_pl, max_placeholder),
                max_universe: core::cmp::max(max_u, max_universe),
            },
        }
    }

    /// If we have reached placeholders, determine if they can
    /// be named from this universe.
    fn can_be_named_by(&self, from: UniverseIndex) -> bool {
        if let PlaceholderReachability::Placeholders { max_universe: (max_universe, _), .. } = self
        {
            from.can_name(*max_universe)
        } else {
            true // No placeholders, no problems.
        }
    }
}

/// An annotation for region graph SCCs that tracks
/// the values of its elements. This annotates a single SCC.
#[derive(Copy, Debug, Clone)]
pub(crate) struct RegionTracker {
    reachable_placeholders: PlaceholderReachability,

    /// The largest universe nameable from this SCC.
    /// It is the smallest nameable universes of all
    /// existential regions reachable from it. Earlier regions in the constraint graph are
    /// preferred.
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
    /// largest nameable uninverse of any reachable region.
    pub(crate) fn max_nameable_universe(self) -> UniverseIndex {
        self.max_nameable_universe.0
    }

    /// Determine if the tracked universes of the two SCCs are compatible.
    pub(crate) fn universe_compatible_with(&self, other: Self) -> bool {
        self.max_nameable_universe().can_name(other.max_nameable_universe())
            || other.reachable_placeholders.can_be_named_by(self.max_nameable_universe())
    }

    /// If this SCC reaches a placeholder it can't name, return it.
    fn unnameable_placeholder(&self) -> Option<(RegionVid, UniverseIndex)> {
        let PlaceholderReachability::Placeholders { max_universe: (max_u, max_u_rvid), .. } =
            self.reachable_placeholders
        else {
            return None;
        };

        if self.max_nameable_universe().can_name(max_u) {
            return None;
        }

        Some((max_u_rvid, max_u))
    }

    /// Check for the second and final type of placeholder leak,
    /// where a placeholder `'p` outlives (transitively) an existential `'e`
    /// and `'e` cannot name `'p`. This is sort of a dual of `unnameable_placeholder`;
    /// one of the members of this SCC cannot be named by the SCC.
    ///
    /// Returns *a* culprit (though there may be more than one).
    fn reaches_existential_that_cannot_name_us(&self) -> Option<RegionVid> {
        let Representative::Placeholder(_p) = self.representative else {
            return None;
        };

        let (reachable_lowest_max_u, reachable_lowest_max_u_rvid) = self.max_nameable_universe;

        (!self.reachable_placeholders.can_be_named_by(reachable_lowest_max_u))
            .then_some(reachable_lowest_max_u_rvid)
    }

    /// Determine if this SCC reaches a placeholder that isn't `placeholder_rvid`,
    /// returning it if that is the case. This prefers the placeholder with the
    /// smallest region variable ID.
    fn reaches_other_placeholder(&self, placeholder_rvid: RegionVid) -> Option<RegionVid> {
        match self.reachable_placeholders {
            PlaceholderReachability::NoPlaceholders => None,
            PlaceholderReachability::Placeholders { min_placeholder, max_placeholder, .. }
                if min_placeholder == max_placeholder =>
            {
                None
            }
            PlaceholderReachability::Placeholders { min_placeholder, max_placeholder, .. }
                if min_placeholder == placeholder_rvid =>
            {
                Some(max_placeholder)
            }
            PlaceholderReachability::Placeholders { min_placeholder, .. } => Some(min_placeholder),
        }
    }
}
/// Pick the smallest universe index out of two, preferring
/// the first argument if they are equal.
#[inline(always)]
fn pick_min_max_universe(a: RegionTracker, b: RegionTracker) -> (UniverseIndex, RegionVid) {
    std::cmp::min_by_key(
        a.max_nameable_universe,
        b.max_nameable_universe,
        |x: &(UniverseIndex, RegionVid)| x.0,
    )
}

impl scc::Annotation for RegionTracker {
    fn merge_scc(self, other: Self) -> Self {
        trace!("{:?} << {:?}", self.representative, other.representative);

        Self {
            reachable_placeholders: self.reachable_placeholders.merge(other.reachable_placeholders),
            max_nameable_universe: pick_min_max_universe(self, other),
            representative: self.representative.merge_scc(other.representative),
        }
    }

    fn merge_reached(mut self, other: Self) -> Self {
        let already_has_unnameable_placeholder = self.unnameable_placeholder().is_some();
        self.max_nameable_universe = pick_min_max_universe(self, other);
        // This detail is subtle. We stop early here, because there may be multiple
        // illegally reached regions, but they are not equally good as blame candidates.
        // In general, the ones with the smallest indices of their RegionVids will
        // be the best ones, and those will also be visited first. This code
        // then will suptly prefer a universe violation happening close from where the
        // constraint graph walk started over one that happens later.
        // FIXME: a potential optimisation if this is slow is to reimplement
        // this check as a boolean fuse, since it will idempotently turn
        // true once triggered and never go false again.
        if already_has_unnameable_placeholder {
            debug!("SCC already has an unnameable placeholder; no use looking for more!");
            self
        } else {
            self.reachable_placeholders =
                self.reachable_placeholders.merge(other.reachable_placeholders);
            self
        }
    }
}

/// Determines if the region variable definitions contain
/// placeholders, and compute them for later use.
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
        let origin = match info.origin {
            RegionVariableOrigin::Nll(origin) => origin,
            _ => NllRegionVariableOrigin::Existential { from_forall: false },
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
    errors_buffer: &mut RegionErrors<'tcx>,
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

    let mut scc_annotations = SccAnnotations::init(&definitions);
    let constraint_sccs = compute_sccs(&outlives_constraints, &mut scc_annotations);

    // This code structure is a bit convoluted because it allows for a planned
    // future change where the early return here has a different type of annotation
    // that does much less work.
    if !has_placeholders {
        debug!("No placeholder regions found; skipping rewriting logic!");

        return LoweredConstraints {
            type_tests,
            member_constraints,
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
        &definitions,
    );

    find_placeholder_mismatch_errors(
        &definitions,
        &constraint_sccs,
        &scc_annotations,
        errors_buffer,
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
        member_constraints,
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
    definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
) -> bool {
    // Changed to `true` if we added any constraints and need to
    // recompute SCCs.
    let mut added_constraints = false;

    let annotations = &annotations.scc_to_annotation;
    let constraint_graph: OnceCell<ConstraintGraph<Normal>> = OnceCell::new();

    for scc in sccs.all_sccs() {
        // No point in adding 'static: 'static!
        // This micro-optimisation makes somewhat sense
        // because static outlives *everything*.
        if scc == sccs.scc(fr_static) {
            continue;
        }

        let annotation = annotations[scc];

        let Some((max_u_rvid, max_u)) = annotation.unnameable_placeholder() else {
            continue;
        };

        debug!(
            "Placeholder universe {max_u:?} is too large for its SCC, represented by {:?}",
            annotation.representative
        );
        // We only add one `r: 'static` constraint per SCC, where `r` is the SCC representative.
        // That constraint is annotated with some outlives relation `tries: unnameable` where
        // `unnameable` is unnameable from `tries` and there is a path in the constraint
        // graph between them.
        //
        // We prefer the representative as `tries` in all cases but one: where the problem
        // is that the SCC has had its universe lowered to accomodate some other region and
        // no longer can name its representative. In that case, we blame `r: low_u`, where `low_u`
        // cannot name `r` so that any explanation always starts with the SCC representative.
        let blame_to = if annotation.representative.rvid() == max_u_rvid {
            // The SCC's representative is not nameable from some region
            // that ends up in the SCC.
            let small_universed_rvid = find_region(
                outlives_constraints,
                constraint_graph.get_or_init(|| outlives_constraints.graph(definitions.len())),
                definitions,
                max_u_rvid,
                |r: RegionVid| definitions[r].universe == annotation.max_nameable_universe(),
                fr_static,
            );
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
            category: ConstraintCategory::IllegalPlaceholder(
                annotation.representative.rvid(),
                blame_to,
            ),
            locations: Locations::All(rustc_span::DUMMY_SP),
            span: rustc_span::DUMMY_SP,
            variance_info: VarianceDiagInfo::None,
            from_closure: false,
        });
    }
    added_constraints
}

// FIXME this is at least partially duplicated code to the constraint search in `region_infer`.
/// Find a region matching a predicate in a set of constraints, using BFS.
fn find_region<'tcx>(
    constraints: &OutlivesConstraintSet<'tcx>,
    graph: &ConstraintGraph<Normal>,
    definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
    start_region: RegionVid,
    target_test: impl Fn(RegionVid) -> bool,
    fr_static: RegionVid,
) -> RegionVid {
    #[derive(Clone, PartialEq, Eq, Debug)]
    enum Trace {
        StartRegion,
        NotVisited,
        Visited,
    }

    let graph = RegionGraph::new(constraints, graph, fr_static);

    let mut context = IndexVec::from_elem(Trace::NotVisited, definitions);
    context[start_region] = Trace::StartRegion;

    let mut deque = VecDeque::new();
    deque.push_back(start_region);

    while let Some(r) = deque.pop_front() {
        if target_test(r) {
            return r;
        }

        for sub_region in graph.outgoing_regions(r) {
            if let Trace::NotVisited = context[sub_region] {
                context[sub_region] = Trace::Visited;
                deque.push_back(sub_region);
            }
        }
    }
    // since this function is used exclusively in this module, we know
    // we are only searching for regions we found in the region graph,
    // so if we don't find what we are looking for there's a bug somwehere.
    bug!("Should have found something!");
}

/// Identify errors where placeholders illegally reach other regions, and generate
/// errors stored into `errors_buffer`.
///
/// There are two sources of such errors:
/// 1. A placeholder reaches (possibly transitively) another placeholder.
/// 2. A placeholder `p` reaches (possibly transitively) an existential `e`,
///    where `e` has an allowed maximum universe smaller than `p`'s.
///
/// There are other potential placeholder errors, but those are detected after
/// region inference, since it may apply type tests or member constraints that
/// alter the contents of SCCs and thus can't be detected at this point.
#[instrument(skip(definitions, sccs, annotations, errors_buffer), level = "debug")]
fn find_placeholder_mismatch_errors<'tcx>(
    definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    annotations: &SccAnnotations<'_, '_, RegionTracker>,
    errors_buffer: &mut RegionErrors<'tcx>,
) {
    use NllRegionVariableOrigin::Placeholder;
    for (rvid, definition) in definitions.iter_enumerated() {
        let Placeholder(origin_a) = definition.origin else {
            continue;
        };

        let scc = sccs.scc(rvid);
        let annotation = annotations.scc_to_annotation[scc];

        if let Some(existental_that_cannot_name_rvid) =
            annotation.reaches_existential_that_cannot_name_us()
        {
            errors_buffer.push(RegionErrorKind::PlaceholderOutlivesExistentialThatCannotNameIt {
                longer_fr: rvid,
                existental_that_cannot_name_longer: existental_that_cannot_name_rvid,
                placeholder: origin_a,
            })
        }

        let Some(other_placeholder) = annotation.reaches_other_placeholder(rvid) else {
            trace!("{rvid:?} reaches no other placeholders");
            continue;
        };

        debug!(
            "Placeholder {rvid:?} of SCC {scc:?} reaches other placeholder {other_placeholder:?}"
        );

        // FIXME SURELY there is a neater way to do this?
        let Placeholder(origin_b) = definitions[other_placeholder].origin else {
            unreachable!(
                "Region {rvid:?}, {other_placeholder:?} should be placeholders but aren't!"
            );
        };

        errors_buffer.push(RegionErrorKind::PlaceholderOutlivesPlaceholder {
            rvid_a: rvid,
            rvid_b: other_placeholder,
            origin_a,
            origin_b,
        });
    }
}
