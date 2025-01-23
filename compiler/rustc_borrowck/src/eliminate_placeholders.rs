//! Logic for lowering higher-kinded outlives constraints
//! (with placeholders and universes) and turn them into regular
//! outlives constraints.
//!
//! This logic is provisional and should be removed once the trait
//! solver can handle this kind of constraint.

use std::collections::VecDeque;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::scc::{self, Sccs};
use rustc_index::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_infer::infer::region_constraints::{VarInfos, VerifyBound};
use rustc_middle::ty::{Region, RegionVid, TyCtxt, UniverseIndex};
use tracing::{debug, instrument};

use crate::constraints::graph::{ConstraintGraph, Normal};
use crate::constraints::{ConstraintSccIndex, OutlivesConstraintSet};
use crate::diagnostics::RegionErrorKind;
use crate::member_constraints::MemberConstraintSet;
use crate::region_infer::{RegionDefinition, Representative, TypeTest};
use crate::universal_regions::UniversalRegions;

pub(crate) struct LoweredConstraints<'tcx> {
    pub(crate) type_tests: Vec<TypeTest<'tcx>>,
    pub(crate) sccs: Sccs<RegionVid, ConstraintSccIndex>,
    pub(crate) definitions: IndexVec<RegionVid, RegionDefinition<'tcx>>,
    pub(crate) scc_representatives: IndexVec<ConstraintSccIndex, Representative>,
    pub(crate) member_constraints: MemberConstraintSet<'tcx, ConstraintSccIndex>,
    pub(crate) outlives_constraints: OutlivesConstraintSet<'tcx>,
}

pub(crate) struct SccAnnotations<'d, 'tcx, A: scc::Annotation> {
    pub(crate) scc_to_annotation: IndexVec<ConstraintSccIndex, A>,
    definitions: &'d IndexVec<RegionVid, RegionDefinition<'tcx>>,
}

impl<'d, 'tcx, A: scc::Annotation> SccAnnotations<'d, 'tcx, A> {
    pub(crate) fn init(definitions: &'d IndexVec<RegionVid, RegionDefinition<'tcx>>) -> Self {
        Self { scc_to_annotation: IndexVec::new(), definitions }
    }
}

#[derive(Copy, Debug, Clone)]
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
/// the values of its elements.
#[derive(Copy, Debug, Clone)]
struct RegionTracker {
    /// The representative Region Variable Id for this SCC.
    representative: Representative,

    /// The smallest universe reachable (and its region)
    min_universe: (UniverseIndex, RegionVid),

    // Metadata about reachable placeholders
    reachable_placeholders: PlaceholderReachability,
}

impl scc::Annotation for RegionTracker {
    fn merge_scc(self, other: Self) -> Self {
        debug!("{:?} << {:?}", self.representative, other.representative);
        let min_universe = if other.min_universe.0 < self.min_universe.0 {
            other.min_universe
        } else {
            self.min_universe
        };
        Self {
            reachable_placeholders: self.reachable_placeholders.merge(other.reachable_placeholders),
            min_universe,
            representative: self.representative.merge_scc(other.representative),
        }
    }

    fn merge_reached(mut self, other: Self) -> Self {
        self.reachable_placeholders =
            self.reachable_placeholders.merge(other.reachable_placeholders);
        self
    }
}

impl RegionTracker {
    fn new(representative: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        let universe_and_rvid = (definition.universe, representative);
        let (representative, reachable_placeholders) = {
            match definition.origin {
                NllRegionVariableOrigin::FreeRegion => (
                    Representative::FreeRegion(representative),
                    PlaceholderReachability::NoPlaceholders,
                ),
                NllRegionVariableOrigin::Placeholder(_) => (
                    Representative::Placeholder(representative),
                    PlaceholderReachability::Placeholders {
                        max_universe: universe_and_rvid,
                        min_placeholder: representative,
                        max_placeholder: representative,
                    },
                ),
                NllRegionVariableOrigin::Existential { .. } => (
                    Representative::Existential(representative),
                    PlaceholderReachability::NoPlaceholders,
                ),
            }
        };
        Self { representative, min_universe: universe_and_rvid, reachable_placeholders }
    }

    /// The smallest-indexed universe reachable from and/or in this SCC.
    fn min_universe(self) -> UniverseIndex {
        self.min_universe.0
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

    /// Determine if the tracked universes of the two SCCs
    /// are compatible.
    fn universe_compatible_with(&self, other: RegionTracker) -> bool {
        self.min_universe().can_name(other.min_universe())
            || other.reachable_placeholders.can_be_named_by(self.min_universe())
    }

    fn representative_rvid(&self) -> RegionVid {
        self.representative.rvid()
    }

    fn into_representative(self) -> Representative {
        self.representative
    }

    /// Determine if this SCC is in the root universe.
    fn in_root_universe(self) -> bool {
        self.min_universe() == UniverseIndex::ROOT
    }
}

impl scc::Annotations<RegionVid, ConstraintSccIndex, RegionTracker>
    for SccAnnotations<'_, '_, RegionTracker>
{
    fn new(&self, element: RegionVid) -> RegionTracker {
        RegionTracker::new(element, &self.definitions[element])
    }

    fn annotate_scc(&mut self, scc: ConstraintSccIndex, annotation: RegionTracker) {
        let idx = self.scc_to_annotation.push(annotation);
        assert!(idx == scc);
    }
}

impl scc::Annotations<RegionVid, ConstraintSccIndex, Representative>
    for SccAnnotations<'_, '_, Representative>
{
    fn new(&self, element: RegionVid) -> Representative {
        Representative::new(element, &self.definitions)
    }

    fn annotate_scc(&mut self, scc: ConstraintSccIndex, annotation: Representative) {
        let idx = self.scc_to_annotation.push(annotation);
        assert!(idx == scc);
    }
}

#[instrument(skip(definitions, sccs, annotations), ret, level = "debug")]
fn find_placeholder_mismatch_errors<'tcx>(
    definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    annotations: &SccAnnotations<'_, '_, RegionTracker>,
) -> Vec<RegionErrorKind<'tcx>> {
    use NllRegionVariableOrigin::Placeholder;

    // Note: it's possible to sort this iterator by SCC and get dependency order,
    // which makes it easy to only add only one constraint per future cycle.
    // However, we want to flag all errors we see, probably.
    let placeholders_and_sccs = definitions.iter_enumerated().filter_map(|(rvid, definition)| {
        if matches!(definition.origin, Placeholder { .. }) {
            Some((sccs.scc(rvid), rvid))
        } else {
            None
        }
    });

    placeholders_and_sccs.filter_map(|(scc, rvid)|{
                let annotation = annotations.scc_to_annotation[scc];
                annotation.reaches_other_placeholder(rvid).and_then(|other_placeholder| {

                    debug!(
                        "Placeholder {rvid:?} of SCC {scc:?} reaches other placeholder {other_placeholder:?}"
                    );

                    // FIXME(amandasystems) -- SURELY there is a neater way to do this?
                    let (Placeholder(origin_a), Placeholder(origin_b)) =
                    (definitions[rvid].origin, definitions[other_placeholder].origin)
                else {
                    unreachable!(
                        "Region {rvid:?}, {other_placeholder:?} should be placeholders but aren't!"
                    );
                };

                Some(RegionErrorKind::PlaceholderMismatch {
                    rvid_a: rvid,
                    rvid_b: other_placeholder,
                    origin_a,
                    origin_b,
                })
                })
            }).collect()
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
/// Every constraint added by this method is an
/// internal `IllegalUniverse` constraint.
#[instrument(skip(tcx))]
pub(crate) fn rewrite_higher_kinded_outlives_as_constraints<'tcx>(
    mut outlives_constraints: OutlivesConstraintSet<'tcx>,
    var_infos: &VarInfos,
    universal_regions: &UniversalRegions<'tcx>,
    type_tests: Vec<TypeTest<'tcx>>,
    tcx: TyCtxt<'tcx>,
    member_constraints: MemberConstraintSet<'tcx, RegionVid>,
) -> (LoweredConstraints<'tcx>, Vec<RegionErrorKind<'tcx>>) {
    // Create a RegionDefinition for each inference variable. This happens here because
    // it allows us to sneak in a cheap check for placeholders. Otherwise, its proper home
    // is in `RegionInferenceContext::new()`, probably.
    let (definitions, has_placeholders) = {
        let mut definitions = IndexVec::with_capacity(var_infos.len());
        let mut has_placeholders = false;

        for info in var_infos.iter() {
            let definition = RegionDefinition::new(info);
            has_placeholders |=
                matches!(definition.origin, NllRegionVariableOrigin::Placeholder(_));
            definitions.push(definition);
        }

        // Add external names from universal regions in fun function definitions.
        for (external_name, variable) in universal_regions.named_universal_regions_iter() {
            debug!("region {:?} has external name {:?}", variable, external_name);
            definitions[variable].external_name = Some(external_name);
        }
        (definitions, has_placeholders)
    };

    if !has_placeholders {
        debug!("No placeholder regions found; skipping rewriting logic!");
        let mut annotations = SccAnnotations::init(&definitions);
        let sccs = outlives_constraints.compute_sccs(
            universal_regions.fr_static,
            definitions.len(),
            &mut annotations,
        );
        return (
            LoweredConstraints {
                type_tests,
                member_constraints: member_constraints.into_mapped(
                    |r| sccs.scc(r),
                    |_| true,
                    |_, _| false,
                ),
                sccs,
                scc_representatives: annotations.scc_to_annotation,
                definitions,
                outlives_constraints,
            },
            Vec::default(),
        );
    }

    debug!("Placeholders present; activating placeholder handling logic!");
    let fr_static = universal_regions.fr_static;

    let mut annotations = SccAnnotations::init(&definitions);
    let sccs = outlives_constraints.compute_sccs(fr_static, definitions.len(), &mut annotations);

    let outlives_static =
        rewrite_outlives(&sccs, &annotations, fr_static, &mut outlives_constraints, &definitions);

    let placeholder_errors = find_placeholder_mismatch_errors(&definitions, &sccs, &annotations);

    let (sccs, scc_annotations) = if !outlives_static.is_empty() {
        debug!("The following SCCs had :'static constraints added: {:?}", outlives_static);
        let mut annotations = SccAnnotations::init(&definitions);

        // We changed the constraint set and so must recompute SCCs.
        // Optimisation opportunity: if we can add them incrementally (and that's
        // possible because edges to 'static always only merge SCCs into 'static),
        // we would potentially save a lot of work here.
        (
            outlives_constraints.compute_sccs(fr_static, definitions.len(), &mut annotations),
            annotations.scc_to_annotation,
        )
    } else {
        // If we didn't add any back-edges; no more work needs doing
        debug!("No constraints rewritten!");
        (sccs, annotations.scc_to_annotation)
    };

    // Rewrite universe-violating type tests into outlives 'static while we remember
    // which universes go where.
    let type_tests = type_tests
        .into_iter()
        .map(|type_test| {
            type_test.rewrite_higher_kinded_constraints(
                &sccs,
                &scc_annotations,
                universal_regions,
                tcx,
            )
        })
        .collect();

    let different_universes = |r1, r2| {
        scc_annotations[sccs.scc(r1)].min_universe() != scc_annotations[sccs.scc(r2)].min_universe()
    };

    // FIXME(amandasystems) it's probably better to destroy and recreate the member constraints from scratch.
    let member_constraints = member_constraints.into_mapped(
        |r| sccs.scc(r),
        |r| scc_annotations[sccs.scc(r)].in_root_universe(),
        different_universes,
    );

    let scc_representatives = scc_annotations
        .into_iter()
        .map(|rich_annotation| rich_annotation.into_representative())
        .collect();

    (
        LoweredConstraints {
            type_tests,
            sccs,
            definitions,
            scc_representatives,
            member_constraints,
            outlives_constraints,
        },
        placeholder_errors,
    )
}

fn rewrite_outlives<'tcx>(
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    annotations: &SccAnnotations<'_, '_, RegionTracker>,
    fr_static: RegionVid,
    outlives_constraints: &mut OutlivesConstraintSet<'tcx>,
    definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
) -> FxHashSet<ConstraintSccIndex> {
    // Is this SCC already outliving static directly or transitively?
    let mut outlives_static = FxHashSet::default();

    let mut memoised_constraint_graph: Option<ConstraintGraph<Normal>> = None;

    for scc in sccs.all_sccs() {
        let annotation: RegionTracker = annotations.scc_to_annotation[scc];
        if scc == sccs.scc(fr_static) {
            // No use adding 'static: 'static.
            continue;
        }

        // Figure out if there is a universe violation in this SCC.
        // This can happen in two cases: either one of our placeholders
        // had its universe lowered from reaching a region with a lower universe,
        // (in which case we blame the lower universe's region), or because we reached
        // a larger universe (in which case we blame the larger universe's region).
        let PlaceholderReachability::Placeholders { max_universe: (max_u, max_u_rvid), .. } =
            annotation.reachable_placeholders
        else {
            continue;
        };

        let (min_u, _) = annotation.min_universe;

        if min_u.can_name(max_u) {
            continue;
        }

        debug!("Universe {max_u:?} is too large for its SCC!");
        let blame_to = if annotation.representative.rvid() == max_u_rvid {
            // We originally had a large enough universe to fit all our reachable
            // placeholders, but had it lowered because we also absorbed something
            // small-universed. In this case, that's to blame!
            let small_universed_rvid = find_region(
                outlives_constraints,
                memoised_constraint_graph
                    .get_or_insert_with(|| outlives_constraints.graph(definitions.len())),
                definitions,
                max_u_rvid,
                |r: RegionVid| definitions[r].universe == min_u,
                fr_static,
            );
            debug!("{small_universed_rvid:?} lowered our universe to {min_u:?}");
            small_universed_rvid
        } else {
            // The problem is that we, who have a small universe, reach a large one.
            max_u_rvid
        };

        outlives_static.insert(scc);
        outlives_constraints.add_placeholder_violation_constraint(
            annotation.representative_rvid(),
            annotation.representative_rvid(),
            blame_to,
            fr_static,
        );
    }
    outlives_static
}

/// Find a region matching a predicate in a set of constraints, using BFS.
// FIXME(amandasystems) this is at least partially duplicated code to the constraint search in `region_infer`.
// It's probably also very expensive.
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

    let mut context = IndexVec::from_elem(Trace::NotVisited, definitions);
    context[start_region] = Trace::StartRegion;

    let mut deque = VecDeque::new();
    deque.push_back(start_region);

    while let Some(r) = deque.pop_front() {
        if target_test(r) {
            return r;
        }

        let outgoing_edges_from_graph = graph.outgoing_edges(r, constraints, Some(fr_static));

        for constraint in outgoing_edges_from_graph {
            debug_assert_eq!(constraint.sup, r);
            let sub_region = constraint.sub;
            if let Trace::NotVisited = context[sub_region] {
                context[sub_region] = Trace::Visited;
                deque.push_back(sub_region);
            }
        }
    }
    unreachable!("Should have found something!");
}

#[instrument(skip(sccs, scc_annotations, universal_regions), ret)]
fn bound_has_universe_violation<'t>(
    bound: &VerifyBound<'t>,
    lower_scc: ConstraintSccIndex,
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    scc_annotations: &IndexVec<ConstraintSccIndex, RegionTracker>,
    universal_regions: &UniversalRegions<'t>,
) -> bool {
    let lower = scc_annotations[lower_scc];
    match bound {
        // An outlives constraint is equivalent to requiring the universe of that region.
        VerifyBound::OutlivedBy(region) => {
            let rvid = universal_regions.to_region_vid(*region);
            if rvid == universal_regions.fr_static {
                false
            } else {
                let bound = scc_annotations[sccs.scc(rvid)];
                !bound.universe_compatible_with(lower)
            }
        }
        // This one is not obvious, but the argument goes something like this:
        // equality is implemented in a later check as verifying that they
        // are in the same SCC. If they are, then they share universe,
        // and so cannot have a universe violation. So this check is strictly
        // weaker than the later SCC membership check, and thus unnecessary.
        VerifyBound::IfEq(_) => false,
        VerifyBound::IsEmpty => false,
        // If all of them have universe violations, this one has too.
        VerifyBound::AnyBound(terms) => {
            !terms.is_empty()
                && terms.iter().all(|t| {
                    bound_has_universe_violation(
                        t,
                        lower_scc,
                        sccs,
                        scc_annotations,
                        universal_regions,
                    )
                })
        }
        // If any of them has a universe violation, this one does.
        VerifyBound::AllBounds(terms) => terms.iter().any(|t| {
            bound_has_universe_violation(t, lower_scc, sccs, scc_annotations, universal_regions)
        }),
    }
}

impl<'t> TypeTest<'t> {
    #[instrument(skip(sccs, tcx, universal_regions, scc_annotations), ret)]
    fn rewrite_higher_kinded_constraints(
        self,
        sccs: &Sccs<RegionVid, ConstraintSccIndex>,
        scc_annotations: &IndexVec<ConstraintSccIndex, RegionTracker>,
        universal_regions: &UniversalRegions<'t>,
        tcx: TyCtxt<'t>,
    ) -> Self {
        let lower_scc = sccs.scc(self.lower_bound);

        if bound_has_universe_violation(
            &self.verify_bound,
            lower_scc,
            sccs,
            scc_annotations,
            universal_regions,
        ) {
            debug!(
                "sub universe `{lower_scc:?}` is not nameable \
                by bound `{:?}`, promoting to 'static",
                self.verify_bound
            );
            let lower_bound = Region::new_var(tcx, self.lower_bound);
            Self {
                lower_bound: universal_regions.fr_static,
                verify_bound: VerifyBound::OutlivedBy(lower_bound),
                original: Some(Box::new(self.clone())),
                ..self
            }
        } else {
            self
        }
    }
}
