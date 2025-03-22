//! Logic for lowering higher-kinded outlives constraints
//! (with placeholders and universes) and turn them into regular
//! outlives constraints.
//!
//! This logic is provisional and should be removed once the trait
//! solver can handle this kind of constraint.

use std::collections::VecDeque;

use either::Either;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::scc::{self, Sccs};
use rustc_index::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_infer::infer::region_constraints::{GenericKind, VarInfos, VerifyBound};
use rustc_infer::infer::relate::TypeRelation;
use rustc_middle::ty::relate::{self, Relate, RelateResult};
use rustc_middle::ty::{self, Region, RegionVid, Ty, TyCtxt, UniverseIndex};
use rustc_span::Span;
use tracing::{debug, instrument, trace};

use crate::ConstraintCategory;
use crate::constraints::graph::{ConstraintGraph, Normal, RegionGraph};
use crate::constraints::{ConstraintSccIndex, OutlivesConstraintSet};
use crate::consumers::OutlivesConstraint;
use crate::diagnostics::{RegionErrorKind, RegionErrors};
use crate::member_constraints::MemberConstraintSet;
use crate::region_infer::{RegionDefinition, Representative, TypeTest};
use crate::ty::VarianceDiagInfo;
use crate::type_check::Locations;
use crate::universal_regions::UniversalRegions;

pub(crate) struct LoweredConstraints<'tcx> {
    pub(crate) type_tests: Vec<LoweredTypeTest<'tcx>>,
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
        trace!("{:?} << {:?}", self.representative, other.representative);
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
        // This detail is subtle. We stop early here, because there may be multiple
        // illegally reached universes, but they are not equally good as blame candidates.
        // In general, the ones with the smallest indices of their RegionVids will
        // be the best ones, and those will also be visited first. This code
        // then will suptly prefer a universe violation caused by a placeholder
        // with a small RegionVid over one with a large RegionVid.
        // FIXME: a potential optimisation if this is slow is to reimplement
        // this check as a boolean fuse, since it will idempotently turn
        // true once triggered and never go false again.
        if self.reaches_too_large_universe().is_some() {
            debug!("SCC already has a placeholder violation; no use looking for more!");
            self
        } else {
            self.reachable_placeholders =
                self.reachable_placeholders.merge(other.reachable_placeholders);
            self
        }
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

    /// If this SCC reaches an universe that's too large, return it.
    fn reaches_too_large_universe(&self) -> Option<(RegionVid, UniverseIndex)> {
        let min_u = self.min_universe();

        let PlaceholderReachability::Placeholders { max_universe: (max_u, max_u_rvid), .. } =
            self.reachable_placeholders
        else {
            return None;
        };

        if min_u.can_name(max_u) {
            return None;
        }

        Some((max_u_rvid, max_u))
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

        let Some(other_placeholder) =
            annotations.scc_to_annotation[scc].reaches_other_placeholder(rvid)
        else {
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

        errors_buffer.push(RegionErrorKind::PlaceholderMismatch {
            rvid_a: rvid,
            rvid_b: other_placeholder,
            origin_a,
            origin_b,
        });
    }
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
#[instrument(skip(tcx, outlives_constraints))]
pub(crate) fn rewrite_higher_kinded_outlives_as_constraints<'tcx>(
    mut outlives_constraints: OutlivesConstraintSet<'tcx>,
    var_infos: &VarInfos,
    universal_regions: &UniversalRegions<'tcx>,
    type_tests: Vec<TypeTest<'tcx>>,
    tcx: TyCtxt<'tcx>,
    member_constraints: MemberConstraintSet<'tcx, RegionVid>,
    errors_buffer: &mut RegionErrors<'tcx>,
) -> LoweredConstraints<'tcx> {
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
        return LoweredConstraints {
            type_tests: type_tests
                .into_iter()
                .map(|type_test| LoweredTypeTest::Untouched(type_test))
                .collect(),
            member_constraints: member_constraints.into_mapped(
                |r| sccs.scc(r),
                |_| true,
                |_, _| false,
            ),
            sccs,
            scc_representatives: annotations.scc_to_annotation,
            definitions,
            outlives_constraints,
        };
    }

    debug!("Placeholders present; activating placeholder handling logic!");
    let fr_static = universal_regions.fr_static;

    let mut annotations = SccAnnotations::init(&definitions);
    let sccs = outlives_constraints.compute_sccs(fr_static, definitions.len(), &mut annotations);

    let outlives_static =
        rewrite_outlives(&sccs, &annotations, fr_static, &mut outlives_constraints, &definitions);

    find_placeholder_mismatch_errors(&definitions, &sccs, &annotations, errors_buffer);

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

    let member_constraints = member_constraints.into_mapped(
        |r| sccs.scc(r),
        |r| scc_annotations[sccs.scc(r)].in_root_universe(),
        different_universes,
    );

    let scc_representatives = scc_annotations
        .into_iter()
        .map(|rich_annotation| rich_annotation.into_representative())
        .collect();

    LoweredConstraints {
        type_tests,
        sccs,
        definitions,
        scc_representatives,
        member_constraints,
        outlives_constraints,
    }
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
        let Some((max_u_rvid, max_u)) = annotation.reaches_too_large_universe() else {
            continue;
        };

        let min_u = annotation.min_universe();

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
        outlives_constraints.push(OutlivesConstraint {
            sup: annotation.representative_rvid(),
            sub: fr_static,
            category: ConstraintCategory::IllegalPlaceholder(
                annotation.representative_rvid(),
                blame_to,
            ),
            locations: Locations::All(rustc_span::DUMMY_SP),
            span: rustc_span::DUMMY_SP,
            variance_info: VarianceDiagInfo::None,
            from_closure: false,
        });
    }
    outlives_static
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
    unreachable!("Should have found something!");
}

#[derive(Debug, Clone)]
pub(crate) enum RewrittenVerifyBound<'tcx> {
    AnyBound(Vec<Either<VerifyBound<'tcx>, RewrittenVerifyBound<'tcx>>>),
    AllBounds(Vec<Either<VerifyBound<'tcx>, RewrittenVerifyBound<'tcx>>>),

    /// Given a region `R`, true if `R: 'static`. Used during lowering
    /// of higher-kinded constraints.
    OutlivesStatic(Region<'tcx>),

    /// The bottom bound, which is never satisfied.
    Unsatisfied,
}

#[derive(Debug, Clone)]
pub(crate) enum LoweredTypeTest<'tcx> {
    Untouched(TypeTest<'tcx>),
    Rewritten {
        rewritten_bound: RewrittenVerifyBound<'tcx>,
        /// The type `T` that must outlive the region.
        generic_kind: GenericKind<'tcx>,

        /// The region `'x` that the type must outlive.
        lower_bound: RegionVid,

        /// The span to blame.
        span: Span,
    },
}

#[instrument(skip(sccs, scc_annotations, universal_regions, tcx), ret)]
fn rewrite_verify_bound<'t>(
    bound: VerifyBound<'t>,
    lower_scc: ConstraintSccIndex,
    sccs: &Sccs<RegionVid, ConstraintSccIndex>,
    scc_annotations: &IndexVec<ConstraintSccIndex, RegionTracker>,
    universal_regions: &UniversalRegions<'t>,
    tcx: TyCtxt<'t>,
    generic_kind: GenericKind<'t>,
) -> Either<VerifyBound<'t>, RewrittenVerifyBound<'t>> {
    let lower = scc_annotations[lower_scc];
    match bound {
        // You may think that an equality bound would imply universe
        // equality, and it does -- except that we do not track placeholders,
        // and so in the event that you have two empty regions, one of which is
        // in an unnameable universe, they would compare equal since they
        // are both empty. This bit ensures that whatever comes out of the
        // bound also matches the placeholder reachability of the lower bound.
        VerifyBound::IfEq(verify_if_eq_b) => {
            let mut m = MatchUniverses::new(tcx, sccs, scc_annotations, universal_regions);
            let verify_if_eq = verify_if_eq_b.skip_binder();
            let what_error = m.relate(verify_if_eq.ty, generic_kind.to_ty(tcx));
            if let Err(e) = what_error {
                debug!("Type test {verify_if_eq_b:?} {generic_kind:?} failed to match with {e:?}");
                return Either::Right(RewrittenVerifyBound::Unsatisfied);
            }

            let r = if let ty::RegionKind::ReBound(depth, _) = verify_if_eq.bound.kind() {
                assert!(depth == ty::INNERMOST);
                m.max_universe_region.map_or(tcx.lifetimes.re_static, |pair| pair.1)
            } else {
                verify_if_eq.bound
            };

            let l_scc = scc_annotations[lower_scc];
            let rvid = universal_regions.to_region_vid(r);

            if rvid == universal_regions.fr_static
                || scc_annotations[sccs.scc(rvid)].universe_compatible_with(l_scc)
            {
                Either::Left(bound)
            } else {
                Either::Right(RewrittenVerifyBound::Unsatisfied)
            }
        }
        // Rewrite an outlives bound to an outlives-static bound upon referencing
        // an unnameable universe (from a placeholder).
        VerifyBound::OutlivedBy(region) => {
            let rvid = universal_regions.to_region_vid(region);
            if rvid == universal_regions.fr_static
                || scc_annotations[sccs.scc(rvid)].universe_compatible_with(lower)
            {
                either::Left(bound)
            } else {
                Either::Right(RewrittenVerifyBound::OutlivesStatic(region))
            }
        }
        // Nothing in here to violate a universe, but since we can't detect
        // bounds being violated by placeholders when we don't track placeholders,
        // we ensure that we don't reach any.
        VerifyBound::IsEmpty => {
            if matches!(lower.reachable_placeholders, PlaceholderReachability::NoPlaceholders) {
                Either::Left(bound)
            } else {
                debug!("Empty bound reaches placeholders: {:?}", lower.reachable_placeholders);
                Either::Right(RewrittenVerifyBound::Unsatisfied)
            }
        }
        // Note that this (and below) claims the contents are always rewritten, even if they weren't.
        // This does not affect correctness, but is surprising.
        VerifyBound::AnyBound(verify_bounds) => Either::Right(RewrittenVerifyBound::AnyBound(
            verify_bounds
                .into_iter()
                .map(|bound| {
                    rewrite_verify_bound(
                        bound,
                        lower_scc,
                        sccs,
                        scc_annotations,
                        universal_regions,
                        tcx,
                        generic_kind,
                    )
                })
                .collect(),
        )),
        VerifyBound::AllBounds(verify_bounds) => Either::Right(RewrittenVerifyBound::AllBounds(
            verify_bounds
                .into_iter()
                .map(|bound| {
                    rewrite_verify_bound(
                        bound,
                        lower_scc,
                        sccs,
                        scc_annotations,
                        universal_regions,
                        tcx,
                        generic_kind,
                    )
                })
                .collect(),
        )),
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
    ) -> LoweredTypeTest<'t> {
        let lower_scc = sccs.scc(self.lower_bound);
        match rewrite_verify_bound(
            self.verify_bound,
            lower_scc,
            sccs,
            scc_annotations,
            universal_regions,
            tcx,
            self.generic_kind,
        ) {
            Either::Left(untouched_bound) => {
                LoweredTypeTest::Untouched(TypeTest { verify_bound: untouched_bound, ..self })
            }
            Either::Right(rewritten_bound) => LoweredTypeTest::Rewritten {
                rewritten_bound,
                generic_kind: self.generic_kind,
                lower_bound: self.lower_bound,
                span: self.span,
            },
        }
    }
}

impl<'tcx, 'v> TypeRelation<TyCtxt<'tcx>> for MatchUniverses<'tcx, 'v> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[instrument(level = "trace", skip(self))]
    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        _: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        // Opaque types args have lifetime parameters.
        // We must not check them to be equal, as we never insert anything to make them so.
        if variance != ty::Bivariant { self.relate(a, b) } else { Ok(a) }
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(
        &mut self,
        pattern: ty::Region<'tcx>,
        value: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        // `pattern` is from inside `VerifyBound::IfEq`, and `value` from `generic_kind` (what we're looking for).
        if pattern == value {
            self.update_max_universe(pattern);
        } else {
            assert!(
                pattern.is_bound() || self.universe_of(pattern).is_root(),
                "{pattern:?} neither bound nor in root universe. Universe is: {:?}, kind: {:?}",
                self.universe_of(pattern),
                pattern.kind()
            );
        }

        if let Some((_, max_universed_region)) = self.max_universe_region.as_ref() {
            Ok(*max_universed_region)
        } else {
            Ok(pattern)
        }
    }

    #[instrument(skip(self), level = "trace")]
    fn tys(&mut self, pattern: Ty<'tcx>, value: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        relate::structurally_relate_tys(self, pattern, value)
    }

    #[instrument(skip(self), level = "trace")]
    fn consts(
        &mut self,
        pattern: ty::Const<'tcx>,
        value: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        relate::structurally_relate_consts(self, pattern, value)
    }

    #[instrument(skip(self), level = "trace")]
    fn binders<T>(
        &mut self,
        pattern: ty::Binder<'tcx, T>,
        value: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        self.pattern_depth.shift_in(1);
        let result = Ok(pattern.rebind(self.relate(pattern.skip_binder(), value.skip_binder())?));
        self.pattern_depth.shift_out(1);
        result
    }
}

struct MatchUniverses<'tcx, 'v> {
    tcx: TyCtxt<'tcx>,
    pattern_depth: ty::DebruijnIndex,
    max_universe_region: Option<(UniverseIndex, ty::Region<'tcx>)>,
    sccs: &'v Sccs<RegionVid, ConstraintSccIndex>,
    scc_annotations: &'v IndexVec<ConstraintSccIndex, RegionTracker>,
    universal_regions: &'v UniversalRegions<'tcx>,
}

impl<'tcx, 'v> MatchUniverses<'tcx, 'v> {
    fn new(
        tcx: TyCtxt<'tcx>,
        sccs: &'v Sccs<RegionVid, ConstraintSccIndex>,
        scc_annotations: &'v IndexVec<ConstraintSccIndex, RegionTracker>,
        universal_regions: &'v UniversalRegions<'tcx>,
    ) -> MatchUniverses<'tcx, 'v> {
        MatchUniverses {
            tcx,
            pattern_depth: ty::INNERMOST,
            max_universe_region: None,
            scc_annotations,
            sccs,
            universal_regions,
        }
    }

    fn universe_of(&self, r: ty::Region<'tcx>) -> UniverseIndex {
        self.scc_annotations[self.sccs.scc(self.universal_regions.to_region_vid(r))].min_universe()
    }

    #[instrument(skip(self), level = "trace")]
    fn update_max_universe(&mut self, r: ty::Region<'tcx>) {
        let r_universe = self.universe_of(r);

        let Some((current_max_u, current_max_r)) = self.max_universe_region else {
            self.max_universe_region = Some((r_universe, r));
            return;
        };
        self.max_universe_region = if r_universe > current_max_u {
            Some((r_universe, r))
        } else {
            Some((current_max_u, current_max_r))
        }
    }
}
