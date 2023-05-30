use std::collections::VecDeque;
use std::rc::Rc;

use rustc_data_structures::binary_search_util;
use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::scc::Sccs;
use rustc_errors::Diagnostic;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_index::{IndexSlice, IndexVec};
use rustc_infer::infer::outlives::test_type_match;
use rustc_infer::infer::region_constraints::{GenericKind, VarInfos, VerifyBound, VerifyIfEq};
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin, RegionVariableOrigin};
use rustc_middle::mir::{
    BasicBlock, Body, ClosureOutlivesRequirement, ClosureOutlivesSubject, ClosureOutlivesSubjectTy,
    ClosureRegionRequirements, ConstraintCategory, Local, Location, ReturnConstraint,
    TerminatorKind,
};
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt, TypeFoldable, TypeVisitableExt};
use rustc_span::Span;

use crate::{
    constraints::{
        graph::NormalConstraintGraph, ConstraintSccIndex, OutlivesConstraint, OutlivesConstraintSet,
    },
    diagnostics::{RegionErrorKind, RegionErrors, UniverseInfo},
    member_constraints::{MemberConstraintSet, NllMemberConstraintIndex},
    nll::PoloniusOutput,
    region_infer::reverse_sccs::ReverseSccGraph,
    region_infer::values::{
        LivenessValues, PlaceholderIndices, RegionElement, RegionValueElements, RegionValues,
        ToElementIndex,
    },
    type_check::{free_region_relations::UniversalRegionRelations, Locations},
    universal_regions::UniversalRegions,
    BorrowckInferCtxt,
};

mod dump_mir;
mod graphviz;
mod opaque_types;
mod reverse_sccs;

pub mod values;

pub struct RegionInferenceContext<'tcx> {
    pub var_infos: VarInfos,

    /// Contains the definition for every region variable. Region
    /// variables are identified by their index (`RegionVid`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    definitions: IndexVec<RegionVid, RegionDefinition<'tcx>>,

    /// The liveness constraints added to each region. For most
    /// regions, these start out empty and steadily grow, though for
    /// each universally quantified region R they start out containing
    /// the entire CFG and `end(R)`.
    liveness_constraints: LivenessValues<RegionVid>,

    /// The outlives constraints computed by the type-check.
    constraints: Frozen<OutlivesConstraintSet<'tcx>>,

    /// The constraint-set, but in graph form, making it easy to traverse
    /// the constraints adjacent to a particular region. Used to construct
    /// the SCC (see `constraint_sccs`) and for error reporting.
    constraint_graph: Frozen<NormalConstraintGraph>,

    /// The SCC computed from `constraints` and the constraint
    /// graph. We have an edge from SCC A to SCC B if `A: B`. Used to
    /// compute the values of each region.
    constraint_sccs: Rc<Sccs<RegionVid, ConstraintSccIndex>>,

    /// Reverse of the SCC constraint graph --  i.e., an edge `A -> B` exists if
    /// `B: A`. This is used to compute the universal regions that are required
    /// to outlive a given SCC. Computed lazily.
    rev_scc_graph: Option<ReverseSccGraph>,

    /// The "R0 member of [R1..Rn]" constraints, indexed by SCC.
    member_constraints: Rc<MemberConstraintSet<'tcx, ConstraintSccIndex>>,

    /// Records the member constraints that we applied to each scc.
    /// This is useful for error reporting. Once constraint
    /// propagation is done, this vector is sorted according to
    /// `member_region_scc`.
    member_constraints_applied: Vec<AppliedMemberConstraint>,

    /// Map universe indexes to information on why we created it.
    universe_causes: FxIndexMap<ty::UniverseIndex, UniverseInfo<'tcx>>,

    /// Contains the minimum universe of any variable within the same
    /// SCC. We will ensure that no SCC contains values that are not
    /// visible from this index.
    scc_universes: IndexVec<ConstraintSccIndex, ty::UniverseIndex>,

    /// Contains a "representative" from each SCC. This will be the
    /// minimal RegionVid belonging to that universe. It is used as a
    /// kind of hacky way to manage checking outlives relationships,
    /// since we can 'canonicalize' each region to the representative
    /// of its SCC and be sure that -- if they have the same repr --
    /// they *must* be equal (though not having the same repr does not
    /// mean they are unequal).
    scc_representatives: IndexVec<ConstraintSccIndex, ty::RegionVid>,

    /// The final inferred values of the region variables; we compute
    /// one value per SCC. To get the value for any given *region*,
    /// you first find which scc it is a part of.
    scc_values: RegionValues<ConstraintSccIndex>,

    /// Type constraints that we check after solving.
    type_tests: Vec<TypeTest<'tcx>>,

    /// Information about the universally quantified regions in scope
    /// on this function.
    universal_regions: Rc<UniversalRegions<'tcx>>,

    /// Information about how the universally quantified regions in
    /// scope on this function relate to one another.
    universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
}

/// Each time that `apply_member_constraint` is successful, it appends
/// one of these structs to the `member_constraints_applied` field.
/// This is used in error reporting to trace out what happened.
///
/// The way that `apply_member_constraint` works is that it effectively
/// adds a new lower bound to the SCC it is analyzing: so you wind up
/// with `'R: 'O` where `'R` is the pick-region and `'O` is the
/// minimal viable option.
#[derive(Debug)]
pub(crate) struct AppliedMemberConstraint {
    /// The SCC that was affected. (The "member region".)
    ///
    /// The vector if `AppliedMemberConstraint` elements is kept sorted
    /// by this field.
    pub(crate) member_region_scc: ConstraintSccIndex,

    /// The "best option" that `apply_member_constraint` found -- this was
    /// added as an "ad-hoc" lower-bound to `member_region_scc`.
    pub(crate) min_choice: ty::RegionVid,

    /// The "member constraint index" -- we can find out details about
    /// the constraint from
    /// `set.member_constraints[member_constraint_index]`.
    pub(crate) member_constraint_index: NllMemberConstraintIndex,
}

pub(crate) struct RegionDefinition<'tcx> {
    /// What kind of variable is this -- a free region? existential
    /// variable? etc. (See the `NllRegionVariableOrigin` for more
    /// info.)
    pub(crate) origin: NllRegionVariableOrigin,

    /// Which universe is this region variable defined in? This is
    /// most often `ty::UniverseIndex::ROOT`, but when we encounter
    /// forall-quantifiers like `for<'a> { 'a = 'b }`, we would create
    /// the variable for `'a` in a fresh universe that extends ROOT.
    pub(crate) universe: ty::UniverseIndex,

    /// If this is 'static or an early-bound region, then this is
    /// `Some(X)` where `X` is the name of the region.
    pub(crate) external_name: Option<ty::Region<'tcx>>,
}

/// N.B., the variants in `Cause` are intentionally ordered. Lower
/// values are preferred when it comes to error messages. Do not
/// reorder willy nilly.
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum Cause {
    /// point inserted because Local was live at the given Location
    LiveVar(Local, Location),

    /// point inserted because Local was dropped at the given Location
    DropVar(Local, Location),
}

/// A "type test" corresponds to an outlives constraint between a type
/// and a lifetime, like `T: 'x` or `<T as Foo>::Bar: 'x`. They are
/// translated from the `Verify` region constraints in the ordinary
/// inference context.
///
/// These sorts of constraints are handled differently than ordinary
/// constraints, at least at present. During type checking, the
/// `InferCtxt::process_registered_region_obligations` method will
/// attempt to convert a type test like `T: 'x` into an ordinary
/// outlives constraint when possible (for example, `&'a T: 'b` will
/// be converted into `'a: 'b` and registered as a `Constraint`).
///
/// In some cases, however, there are outlives relationships that are
/// not converted into a region constraint, but rather into one of
/// these "type tests". The distinction is that a type test does not
/// influence the inference result, but instead just examines the
/// values that we ultimately inferred for each region variable and
/// checks that they meet certain extra criteria. If not, an error
/// can be issued.
///
/// One reason for this is that these type tests typically boil down
/// to a check like `'a: 'x` where `'a` is a universally quantified
/// region -- and therefore not one whose value is really meant to be
/// *inferred*, precisely (this is not always the case: one can have a
/// type test like `<Foo as Trait<'?0>>::Bar: 'x`, where `'?0` is an
/// inference variable). Another reason is that these type tests can
/// involve *disjunction* -- that is, they can be satisfied in more
/// than one way.
///
/// For more information about this translation, see
/// `InferCtxt::process_registered_region_obligations` and
/// `InferCtxt::type_must_outlive` in `rustc_infer::infer::InferCtxt`.
#[derive(Clone, Debug)]
pub struct TypeTest<'tcx> {
    /// The type `T` that must outlive the region.
    pub generic_kind: GenericKind<'tcx>,

    /// The region `'x` that the type must outlive.
    pub lower_bound: RegionVid,

    /// The span to blame.
    pub span: Span,

    /// A test which, if met by the region `'x`, proves that this type
    /// constraint is satisfied.
    pub verify_bound: VerifyBound<'tcx>,
}

/// When we have an unmet lifetime constraint, we try to propagate it outward (e.g. to a closure
/// environment). If we can't, it is an error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegionRelationCheckResult {
    Ok,
    Propagated,
    Error,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Trace<'tcx> {
    StartRegion,
    FromOutlivesConstraint(OutlivesConstraint<'tcx>),
    NotVisited,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExtraConstraintInfo {
    PlaceholderFromPredicate(Span),
}

#[instrument(skip(infcx, sccs), level = "debug")]
fn sccs_info<'cx, 'tcx>(
    infcx: &'cx BorrowckInferCtxt<'cx, 'tcx>,
    sccs: Rc<Sccs<RegionVid, ConstraintSccIndex>>,
) {
    use crate::renumber::RegionCtxt;

    let var_to_origin = infcx.reg_var_to_origin.borrow();

    let mut var_to_origin_sorted = var_to_origin.clone().into_iter().collect::<Vec<_>>();
    var_to_origin_sorted.sort_by_key(|vto| vto.0);

    let mut reg_vars_to_origins_str = "region variables to origins:\n".to_string();
    for (reg_var, origin) in var_to_origin_sorted.into_iter() {
        reg_vars_to_origins_str.push_str(&format!("{:?}: {:?}\n", reg_var, origin));
    }
    debug!("{}", reg_vars_to_origins_str);

    let num_components = sccs.scc_data().ranges().len();
    let mut components = vec![FxIndexSet::default(); num_components];

    for (reg_var_idx, scc_idx) in sccs.scc_indices().iter().enumerate() {
        let reg_var = ty::RegionVid::from_usize(reg_var_idx);
        let origin = var_to_origin.get(&reg_var).unwrap_or_else(|| &RegionCtxt::Unknown);
        components[scc_idx.as_usize()].insert((reg_var, *origin));
    }

    let mut components_str = "strongly connected components:".to_string();
    for (scc_idx, reg_vars_origins) in components.iter().enumerate() {
        let regions_info = reg_vars_origins.clone().into_iter().collect::<Vec<_>>();
        components_str.push_str(&format!(
            "{:?}: {:?},\n)",
            ConstraintSccIndex::from_usize(scc_idx),
            regions_info,
        ))
    }
    debug!("{}", components_str);

    // calculate the best representative for each component
    let components_representatives = components
        .into_iter()
        .enumerate()
        .map(|(scc_idx, region_ctxts)| {
            let repr = region_ctxts
                .into_iter()
                .map(|reg_var_origin| reg_var_origin.1)
                .max_by(|x, y| x.preference_value().cmp(&y.preference_value()))
                .unwrap();

            (ConstraintSccIndex::from_usize(scc_idx), repr)
        })
        .collect::<FxIndexMap<_, _>>();

    let mut scc_node_to_edges = FxIndexMap::default();
    for (scc_idx, repr) in components_representatives.iter() {
        let edges_range = sccs.scc_data().ranges()[*scc_idx].clone();
        let edges = &sccs.scc_data().all_successors()[edges_range];
        let edge_representatives =
            edges.iter().map(|scc_idx| components_representatives[scc_idx]).collect::<Vec<_>>();
        scc_node_to_edges.insert((scc_idx, repr), edge_representatives);
    }

    debug!("SCC edges {:#?}", scc_node_to_edges);
}

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Creates a new region inference context with a total of
    /// `num_region_variables` valid inference variables; the first N
    /// of those will be constant regions representing the free
    /// regions defined in `universal_regions`.
    ///
    /// The `outlives_constraints` and `type_tests` are an initial set
    /// of constraints produced by the MIR type check.
    pub(crate) fn new<'cx>(
        _infcx: &BorrowckInferCtxt<'cx, 'tcx>,
        var_infos: VarInfos,
        universal_regions: Rc<UniversalRegions<'tcx>>,
        placeholder_indices: Rc<PlaceholderIndices>,
        universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
        outlives_constraints: OutlivesConstraintSet<'tcx>,
        member_constraints_in: MemberConstraintSet<'tcx, RegionVid>,
        universe_causes: FxIndexMap<ty::UniverseIndex, UniverseInfo<'tcx>>,
        type_tests: Vec<TypeTest<'tcx>>,
        liveness_constraints: LivenessValues<RegionVid>,
        elements: &Rc<RegionValueElements>,
    ) -> Self {
        debug!("universal_regions: {:#?}", universal_regions);
        debug!("outlives constraints: {:#?}", outlives_constraints);
        debug!("placeholder_indices: {:#?}", placeholder_indices);
        debug!("type tests: {:#?}", type_tests);

        // Create a RegionDefinition for each inference variable.
        let definitions: IndexVec<_, _> = var_infos
            .iter()
            .map(|info| RegionDefinition::new(info.universe, info.origin))
            .collect();

        let constraints = Frozen::freeze(outlives_constraints);
        let constraint_graph = Frozen::freeze(constraints.graph(definitions.len()));
        let fr_static = universal_regions.fr_static;
        let constraint_sccs = Rc::new(constraints.compute_sccs(&constraint_graph, fr_static));

        if cfg!(debug_assertions) {
            sccs_info(_infcx, constraint_sccs.clone());
        }

        let mut scc_values =
            RegionValues::new(elements, universal_regions.len(), &placeholder_indices);

        for region in liveness_constraints.rows() {
            let scc = constraint_sccs.scc(region);
            scc_values.merge_liveness(scc, region, &liveness_constraints);
        }

        let scc_universes = Self::compute_scc_universes(&constraint_sccs, &definitions);

        let scc_representatives = Self::compute_scc_representatives(&constraint_sccs, &definitions);

        let member_constraints =
            Rc::new(member_constraints_in.into_mapped(|r| constraint_sccs.scc(r)));

        let mut result = Self {
            var_infos,
            definitions,
            liveness_constraints,
            constraints,
            constraint_graph,
            constraint_sccs,
            rev_scc_graph: None,
            member_constraints,
            member_constraints_applied: Vec::new(),
            universe_causes,
            scc_universes,
            scc_representatives,
            scc_values,
            type_tests,
            universal_regions,
            universal_region_relations,
        };

        result.init_free_and_bound_regions();

        result
    }

    /// Each SCC is the combination of many region variables which
    /// have been equated. Therefore, we can associate a universe with
    /// each SCC which is minimum of all the universes of its
    /// constituent regions -- this is because whatever value the SCC
    /// takes on must be a value that each of the regions within the
    /// SCC could have as well. This implies that the SCC must have
    /// the minimum, or narrowest, universe.
    fn compute_scc_universes(
        constraint_sccs: &Sccs<RegionVid, ConstraintSccIndex>,
        definitions: &IndexSlice<RegionVid, RegionDefinition<'tcx>>,
    ) -> IndexVec<ConstraintSccIndex, ty::UniverseIndex> {
        let num_sccs = constraint_sccs.num_sccs();
        let mut scc_universes = IndexVec::from_elem_n(ty::UniverseIndex::MAX, num_sccs);

        debug!("compute_scc_universes()");

        // For each region R in universe U, ensure that the universe for the SCC
        // that contains R is "no bigger" than U. This effectively sets the universe
        // for each SCC to be the minimum of the regions within.
        for (region_vid, region_definition) in definitions.iter_enumerated() {
            let scc = constraint_sccs.scc(region_vid);
            let scc_universe = &mut scc_universes[scc];
            let scc_min = std::cmp::min(region_definition.universe, *scc_universe);
            if scc_min != *scc_universe {
                *scc_universe = scc_min;
                debug!(
                    "compute_scc_universes: lowered universe of {scc:?} to {scc_min:?} \
                    because it contains {region_vid:?} in {region_universe:?}",
                    scc = scc,
                    scc_min = scc_min,
                    region_vid = region_vid,
                    region_universe = region_definition.universe,
                );
            }
        }

        // Walk each SCC `A` and `B` such that `A: B`
        // and ensure that universe(A) can see universe(B).
        //
        // This serves to enforce the 'empty/placeholder' hierarchy
        // (described in more detail on `RegionKind`):
        //
        // ```
        // static -----+
        //   |         |
        // empty(U0) placeholder(U1)
        //   |      /
        // empty(U1)
        // ```
        //
        // In particular, imagine we have variables R0 in U0 and R1
        // created in U1, and constraints like this;
        //
        // ```
        // R1: !1 // R1 outlives the placeholder in U1
        // R1: R0 // R1 outlives R0
        // ```
        //
        // Here, we wish for R1 to be `'static`, because it
        // cannot outlive `placeholder(U1)` and `empty(U0)` any other way.
        //
        // Thanks to this loop, what happens is that the `R1: R0`
        // constraint lowers the universe of `R1` to `U0`, which in turn
        // means that the `R1: !1` constraint will (later) cause
        // `R1` to become `'static`.
        for scc_a in constraint_sccs.all_sccs() {
            for &scc_b in constraint_sccs.successors(scc_a) {
                let scc_universe_a = scc_universes[scc_a];
                let scc_universe_b = scc_universes[scc_b];
                let scc_universe_min = std::cmp::min(scc_universe_a, scc_universe_b);
                if scc_universe_a != scc_universe_min {
                    scc_universes[scc_a] = scc_universe_min;

                    debug!(
                        "compute_scc_universes: lowered universe of {scc_a:?} to {scc_universe_min:?} \
                        because {scc_a:?}: {scc_b:?} and {scc_b:?} is in universe {scc_universe_b:?}",
                        scc_a = scc_a,
                        scc_b = scc_b,
                        scc_universe_min = scc_universe_min,
                        scc_universe_b = scc_universe_b
                    );
                }
            }
        }

        debug!("compute_scc_universes: scc_universe = {:#?}", scc_universes);

        scc_universes
    }

    /// For each SCC, we compute a unique `RegionVid` (in fact, the
    /// minimal one that belongs to the SCC). See
    /// `scc_representatives` field of `RegionInferenceContext` for
    /// more details.
    fn compute_scc_representatives(
        constraints_scc: &Sccs<RegionVid, ConstraintSccIndex>,
        definitions: &IndexSlice<RegionVid, RegionDefinition<'tcx>>,
    ) -> IndexVec<ConstraintSccIndex, ty::RegionVid> {
        let num_sccs = constraints_scc.num_sccs();
        let next_region_vid = definitions.next_index();
        let mut scc_representatives = IndexVec::from_elem_n(next_region_vid, num_sccs);

        for region_vid in definitions.indices() {
            let scc = constraints_scc.scc(region_vid);
            let prev_min = scc_representatives[scc];
            scc_representatives[scc] = region_vid.min(prev_min);
        }

        scc_representatives
    }

    /// Initializes the region variables for each universally
    /// quantified region (lifetime parameter). The first N variables
    /// always correspond to the regions appearing in the function
    /// signature (both named and anonymous) and where-clauses. This
    /// function iterates over those regions and initializes them with
    /// minimum values.
    ///
    /// For example:
    /// ```
    /// fn foo<'a, 'b>( /* ... */ ) where 'a: 'b { /* ... */ }
    /// ```
    /// would initialize two variables like so:
    /// ```ignore (illustrative)
    /// R0 = { CFG, R0 } // 'a
    /// R1 = { CFG, R0, R1 } // 'b
    /// ```
    /// Here, R0 represents `'a`, and it contains (a) the entire CFG
    /// and (b) any universally quantified regions that it outlives,
    /// which in this case is just itself. R1 (`'b`) in contrast also
    /// outlives `'a` and hence contains R0 and R1.
    fn init_free_and_bound_regions(&mut self) {
        // Update the names (if any)
        // This iterator has unstable order but we collect it all into an IndexVec
        for (external_name, variable) in self.universal_regions.named_universal_regions() {
            debug!(
                "init_universal_regions: region {:?} has external name {:?}",
                variable, external_name
            );
            self.definitions[variable].external_name = Some(external_name);
        }

        for variable in self.definitions.indices() {
            let scc = self.constraint_sccs.scc(variable);

            match self.definitions[variable].origin {
                NllRegionVariableOrigin::FreeRegion => {
                    // For each free, universally quantified region X:

                    // Add all nodes in the CFG to liveness constraints
                    self.liveness_constraints.add_all_points(variable);
                    self.scc_values.add_all_points(scc);

                    // Add `end(X)` into the set for X.
                    self.scc_values.add_element(scc, variable);
                }

                NllRegionVariableOrigin::Placeholder(placeholder) => {
                    // Each placeholder region is only visible from
                    // its universe `ui` and its extensions. So we
                    // can't just add it into `scc` unless the
                    // universe of the scc can name this region.
                    let scc_universe = self.scc_universes[scc];
                    if scc_universe.can_name(placeholder.universe) {
                        self.scc_values.add_element(scc, placeholder);
                    } else {
                        debug!(
                            "init_free_and_bound_regions: placeholder {:?} is \
                             not compatible with universe {:?} of its SCC {:?}",
                            placeholder, scc_universe, scc,
                        );
                        self.add_incompatible_universe(scc);
                    }
                }

                NllRegionVariableOrigin::Existential { .. } => {
                    // For existential, regions, nothing to do.
                }
            }
        }
    }

    /// Returns an iterator over all the region indices.
    pub fn regions(&self) -> impl Iterator<Item = RegionVid> + 'tcx {
        self.definitions.indices()
    }

    /// Given a universal region in scope on the MIR, returns the
    /// corresponding index.
    ///
    /// (Panics if `r` is not a registered universal region.)
    pub fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        self.universal_regions.to_region_vid(r)
    }

    /// Returns an iterator over all the outlives constraints.
    pub fn outlives_constraints(&self) -> impl Iterator<Item = OutlivesConstraint<'tcx>> + '_ {
        self.constraints.outlives().iter().copied()
    }

    /// Adds annotations for `#[rustc_regions]`; see `UniversalRegions::annotate`.
    pub(crate) fn annotate(&self, tcx: TyCtxt<'tcx>, err: &mut Diagnostic) {
        self.universal_regions.annotate(tcx, err)
    }

    /// Returns `true` if the region `r` contains the point `p`.
    ///
    /// Panics if called before `solve()` executes,
    pub(crate) fn region_contains(&self, r: RegionVid, p: impl ToElementIndex) -> bool {
        let scc = self.constraint_sccs.scc(r);
        self.scc_values.contains(scc, p)
    }

    /// Returns the lowest statement index in `start..=end` which is not contained by `r`.
    ///
    /// Panics if called before `solve()` executes.
    pub(crate) fn first_non_contained_inclusive(
        &self,
        r: RegionVid,
        block: BasicBlock,
        start: usize,
        end: usize,
    ) -> Option<usize> {
        let scc = self.constraint_sccs.scc(r);
        self.scc_values.first_non_contained_inclusive(scc, block, start, end)
    }

    /// Returns access to the value of `r` for debugging purposes.
    pub(crate) fn region_value_str(&self, r: RegionVid) -> String {
        let scc = self.constraint_sccs.scc(r);
        self.scc_values.region_value_str(scc)
    }

    pub(crate) fn placeholders_contained_in<'a>(
        &'a self,
        r: RegionVid,
    ) -> impl Iterator<Item = ty::PlaceholderRegion> + 'a {
        let scc = self.constraint_sccs.scc(r);
        self.scc_values.placeholders_contained_in(scc)
    }

    /// Returns access to the value of `r` for debugging purposes.
    pub(crate) fn region_universe(&self, r: RegionVid) -> ty::UniverseIndex {
        let scc = self.constraint_sccs.scc(r);
        self.scc_universes[scc]
    }

    /// Once region solving has completed, this function will return
    /// the member constraints that were applied to the value of a given
    /// region `r`. See `AppliedMemberConstraint`.
    pub(crate) fn applied_member_constraints(&self, r: RegionVid) -> &[AppliedMemberConstraint] {
        let scc = self.constraint_sccs.scc(r);
        binary_search_util::binary_search_slice(
            &self.member_constraints_applied,
            |applied| applied.member_region_scc,
            &scc,
        )
    }

    /// Performs region inference and report errors if we see any
    /// unsatisfiable constraints. If this is a closure, returns the
    /// region requirements to propagate to our creator, if any.
    #[instrument(skip(self, infcx, body, polonius_output), level = "debug")]
    pub(super) fn solve(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body: &Body<'tcx>,
        polonius_output: Option<Rc<PoloniusOutput>>,
    ) -> (Option<ClosureRegionRequirements<'tcx>>, RegionErrors<'tcx>) {
        let mir_def_id = body.source.def_id();
        self.propagate_constraints(body);

        let mut errors_buffer = RegionErrors::new(infcx.tcx);

        // If this is a closure, we can propagate unsatisfied
        // `outlives_requirements` to our creator, so create a vector
        // to store those. Otherwise, we'll pass in `None` to the
        // functions below, which will trigger them to report errors
        // eagerly.
        let mut outlives_requirements = infcx.tcx.is_typeck_child(mir_def_id).then(Vec::new);

        self.check_type_tests(
            infcx,
            param_env,
            body,
            outlives_requirements.as_mut(),
            &mut errors_buffer,
        );

        // In Polonius mode, the errors about missing universal region relations are in the output
        // and need to be emitted or propagated. Otherwise, we need to check whether the
        // constraints were too strong, and if so, emit or propagate those errors.
        if infcx.tcx.sess.opts.unstable_opts.polonius {
            self.check_polonius_subset_errors(
                outlives_requirements.as_mut(),
                &mut errors_buffer,
                polonius_output.expect("Polonius output is unavailable despite `-Z polonius`"),
            );
        } else {
            self.check_universal_regions(outlives_requirements.as_mut(), &mut errors_buffer);
        }

        if errors_buffer.is_empty() {
            self.check_member_constraints(infcx, &mut errors_buffer);
        }

        let outlives_requirements = outlives_requirements.unwrap_or_default();

        if outlives_requirements.is_empty() {
            (None, errors_buffer)
        } else {
            let num_external_vids = self.universal_regions.num_global_and_external_regions();
            (
                Some(ClosureRegionRequirements { num_external_vids, outlives_requirements }),
                errors_buffer,
            )
        }
    }

    /// Propagate the region constraints: this will grow the values
    /// for each region variable until all the constraints are
    /// satisfied. Note that some values may grow **too** large to be
    /// feasible, but we check this later.
    #[instrument(skip(self, _body), level = "debug")]
    fn propagate_constraints(&mut self, _body: &Body<'tcx>) {
        debug!("constraints={:#?}", {
            let mut constraints: Vec<_> = self.outlives_constraints().collect();
            constraints.sort_by_key(|c| (c.sup, c.sub));
            constraints
                .into_iter()
                .map(|c| (c, self.constraint_sccs.scc(c.sup), self.constraint_sccs.scc(c.sub)))
                .collect::<Vec<_>>()
        });

        // To propagate constraints, we walk the DAG induced by the
        // SCC. For each SCC, we visit its successors and compute
        // their values, then we union all those values to get our
        // own.
        let constraint_sccs = self.constraint_sccs.clone();
        for scc in constraint_sccs.all_sccs() {
            self.compute_value_for_scc(scc);
        }

        // Sort the applied member constraints so we can binary search
        // through them later.
        self.member_constraints_applied.sort_by_key(|applied| applied.member_region_scc);
    }

    /// Computes the value of the SCC `scc_a`, which has not yet been
    /// computed, by unioning the values of its successors.
    /// Assumes that all successors have been computed already
    /// (which is assured by iterating over SCCs in dependency order).
    #[instrument(skip(self), level = "debug")]
    fn compute_value_for_scc(&mut self, scc_a: ConstraintSccIndex) {
        let constraint_sccs = self.constraint_sccs.clone();

        // Walk each SCC `B` such that `A: B`...
        for &scc_b in constraint_sccs.successors(scc_a) {
            debug!(?scc_b);

            // ...and add elements from `B` into `A`. One complication
            // arises because of universes: If `B` contains something
            // that `A` cannot name, then `A` can only contain `B` if
            // it outlives static.
            if self.universe_compatible(scc_b, scc_a) {
                // `A` can name everything that is in `B`, so just
                // merge the bits.
                self.scc_values.add_region(scc_a, scc_b);
            } else {
                self.add_incompatible_universe(scc_a);
            }
        }

        // Now take member constraints into account.
        let member_constraints = self.member_constraints.clone();
        for m_c_i in member_constraints.indices(scc_a) {
            self.apply_member_constraint(scc_a, m_c_i, member_constraints.choice_regions(m_c_i));
        }

        debug!(value = ?self.scc_values.region_value_str(scc_a));
    }

    /// Invoked for each `R0 member of [R1..Rn]` constraint.
    ///
    /// `scc` is the SCC containing R0, and `choice_regions` are the
    /// `R1..Rn` regions -- they are always known to be universal
    /// regions (and if that's not true, we just don't attempt to
    /// enforce the constraint).
    ///
    /// The current value of `scc` at the time the method is invoked
    /// is considered a *lower bound*. If possible, we will modify
    /// the constraint to set it equal to one of the option regions.
    /// If we make any changes, returns true, else false.
    #[instrument(skip(self, member_constraint_index), level = "debug")]
    fn apply_member_constraint(
        &mut self,
        scc: ConstraintSccIndex,
        member_constraint_index: NllMemberConstraintIndex,
        choice_regions: &[ty::RegionVid],
    ) -> bool {
        // Create a mutable vector of the options. We'll try to winnow
        // them down.
        let mut choice_regions: Vec<ty::RegionVid> = choice_regions.to_vec();

        // Convert to the SCC representative: sometimes we have inference
        // variables in the member constraint that wind up equated with
        // universal regions. The scc representative is the minimal numbered
        // one from the corresponding scc so it will be the universal region
        // if one exists.
        for c_r in &mut choice_regions {
            let scc = self.constraint_sccs.scc(*c_r);
            *c_r = self.scc_representatives[scc];
        }

        // The 'member region' in a member constraint is part of the
        // hidden type, which must be in the root universe. Therefore,
        // it cannot have any placeholders in its value.
        assert!(self.scc_universes[scc] == ty::UniverseIndex::ROOT);
        debug_assert!(
            self.scc_values.placeholders_contained_in(scc).next().is_none(),
            "scc {:?} in a member constraint has placeholder value: {:?}",
            scc,
            self.scc_values.region_value_str(scc),
        );

        // The existing value for `scc` is a lower-bound. This will
        // consist of some set `{P} + {LB}` of points `{P}` and
        // lower-bound free regions `{LB}`. As each choice region `O`
        // is a free region, it will outlive the points. But we can
        // only consider the option `O` if `O: LB`.
        choice_regions.retain(|&o_r| {
            self.scc_values
                .universal_regions_outlived_by(scc)
                .all(|lb| self.universal_region_relations.outlives(o_r, lb))
        });
        debug!(?choice_regions, "after lb");

        // Now find all the *upper bounds* -- that is, each UB is a
        // free region that must outlive the member region `R0` (`UB:
        // R0`). Therefore, we need only keep an option `O` if `UB: O`
        // for all UB.
        self.compute_reverse_scc_graph();
        let universal_region_relations = &self.universal_region_relations;
        for ub in self.rev_scc_graph.as_ref().unwrap().upper_bounds(scc) {
            debug!(?ub);
            choice_regions.retain(|&o_r| universal_region_relations.outlives(ub, o_r));
        }
        debug!(?choice_regions, "after ub");

        // At this point we can pick any member of `choice_regions`, but to avoid potential
        // non-determinism we will pick the *unique minimum* choice.
        //
        // Because universal regions are only partially ordered (i.e, not every two regions are
        // comparable), we will ignore any region that doesn't compare to all others when picking
        // the minimum choice.
        // For example, consider `choice_regions = ['static, 'a, 'b, 'c, 'd, 'e]`, where
        // `'static: 'a, 'static: 'b, 'a: 'c, 'b: 'c, 'c: 'd, 'c: 'e`.
        // `['d, 'e]` are ignored because they do not compare - the same goes for `['a, 'b]`.
        let totally_ordered_subset = choice_regions.iter().copied().filter(|&r1| {
            choice_regions.iter().all(|&r2| {
                self.universal_region_relations.outlives(r1, r2)
                    || self.universal_region_relations.outlives(r2, r1)
            })
        });
        // Now we're left with `['static, 'c]`. Pick `'c` as the minimum!
        let Some(min_choice) = totally_ordered_subset.reduce(|r1, r2| {
            let r1_outlives_r2 = self.universal_region_relations.outlives(r1, r2);
            let r2_outlives_r1 = self.universal_region_relations.outlives(r2, r1);
            match (r1_outlives_r2, r2_outlives_r1) {
                (true, true) => r1.min(r2),
                (true, false) => r2,
                (false, true) => r1,
                (false, false) => bug!("incomparable regions in total order"),
            }
        }) else {
            debug!("no unique minimum choice");
            return false;
        };

        let min_choice_scc = self.constraint_sccs.scc(min_choice);
        debug!(?min_choice, ?min_choice_scc);
        if self.scc_values.add_region(scc, min_choice_scc) {
            self.member_constraints_applied.push(AppliedMemberConstraint {
                member_region_scc: scc,
                min_choice,
                member_constraint_index,
            });

            true
        } else {
            false
        }
    }

    /// Returns `true` if all the elements in the value of `scc_b` are nameable
    /// in `scc_a`. Used during constraint propagation, and only once
    /// the value of `scc_b` has been computed.
    fn universe_compatible(&self, scc_b: ConstraintSccIndex, scc_a: ConstraintSccIndex) -> bool {
        let universe_a = self.scc_universes[scc_a];

        // Quick check: if scc_b's declared universe is a subset of
        // scc_a's declared universe (typically, both are ROOT), then
        // it cannot contain any problematic universe elements.
        if universe_a.can_name(self.scc_universes[scc_b]) {
            return true;
        }

        // Otherwise, we have to iterate over the universe elements in
        // B's value, and check whether all of them are nameable
        // from universe_a
        self.scc_values.placeholders_contained_in(scc_b).all(|p| universe_a.can_name(p.universe))
    }

    /// Extend `scc` so that it can outlive some placeholder region
    /// from a universe it can't name; at present, the only way for
    /// this to be true is if `scc` outlives `'static`. This is
    /// actually stricter than necessary: ideally, we'd support bounds
    /// like `for<'a: 'b>` that might then allow us to approximate
    /// `'a` with `'b` and not `'static`. But it will have to do for
    /// now.
    fn add_incompatible_universe(&mut self, scc: ConstraintSccIndex) {
        debug!("add_incompatible_universe(scc={:?})", scc);

        let fr_static = self.universal_regions.fr_static;
        self.scc_values.add_all_points(scc);
        self.scc_values.add_element(scc, fr_static);
    }

    /// Once regions have been propagated, this method is used to see
    /// whether the "type tests" produced by typeck were satisfied;
    /// type tests encode type-outlives relationships like `T:
    /// 'a`. See `TypeTest` for more details.
    fn check_type_tests(
        &self,
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body: &Body<'tcx>,
        mut propagated_outlives_requirements: Option<&mut Vec<ClosureOutlivesRequirement<'tcx>>>,
        errors_buffer: &mut RegionErrors<'tcx>,
    ) {
        let tcx = infcx.tcx;

        // Sometimes we register equivalent type-tests that would
        // result in basically the exact same error being reported to
        // the user. Avoid that.
        let mut deduplicate_errors = FxIndexSet::default();

        for type_test in &self.type_tests {
            debug!("check_type_test: {:?}", type_test);

            let generic_ty = type_test.generic_kind.to_ty(tcx);
            if self.eval_verify_bound(
                infcx,
                param_env,
                generic_ty,
                type_test.lower_bound,
                &type_test.verify_bound,
            ) {
                continue;
            }

            if let Some(propagated_outlives_requirements) = &mut propagated_outlives_requirements {
                if self.try_promote_type_test(
                    infcx,
                    param_env,
                    body,
                    type_test,
                    propagated_outlives_requirements,
                ) {
                    continue;
                }
            }

            // Type-test failed. Report the error.
            let erased_generic_kind = infcx.tcx.erase_regions(type_test.generic_kind);

            // Skip duplicate-ish errors.
            if deduplicate_errors.insert((
                erased_generic_kind,
                type_test.lower_bound,
                type_test.span,
            )) {
                debug!(
                    "check_type_test: reporting error for erased_generic_kind={:?}, \
                     lower_bound_region={:?}, \
                     type_test.span={:?}",
                    erased_generic_kind, type_test.lower_bound, type_test.span,
                );

                errors_buffer.push(RegionErrorKind::TypeTestError { type_test: type_test.clone() });
            }
        }
    }

    /// Invoked when we have some type-test (e.g., `T: 'X`) that we cannot
    /// prove to be satisfied. If this is a closure, we will attempt to
    /// "promote" this type-test into our `ClosureRegionRequirements` and
    /// hence pass it up the creator. To do this, we have to phrase the
    /// type-test in terms of external free regions, as local free
    /// regions are not nameable by the closure's creator.
    ///
    /// Promotion works as follows: we first check that the type `T`
    /// contains only regions that the creator knows about. If this is
    /// true, then -- as a consequence -- we know that all regions in
    /// the type `T` are free regions that outlive the closure body. If
    /// false, then promotion fails.
    ///
    /// Once we've promoted T, we have to "promote" `'X` to some region
    /// that is "external" to the closure. Generally speaking, a region
    /// may be the union of some points in the closure body as well as
    /// various free lifetimes. We can ignore the points in the closure
    /// body: if the type T can be expressed in terms of external regions,
    /// we know it outlives the points in the closure body. That
    /// just leaves the free regions.
    ///
    /// The idea then is to lower the `T: 'X` constraint into multiple
    /// bounds -- e.g., if `'X` is the union of two free lifetimes,
    /// `'1` and `'2`, then we would create `T: '1` and `T: '2`.
    #[instrument(level = "debug", skip(self, infcx, propagated_outlives_requirements))]
    fn try_promote_type_test(
        &self,
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        body: &Body<'tcx>,
        type_test: &TypeTest<'tcx>,
        propagated_outlives_requirements: &mut Vec<ClosureOutlivesRequirement<'tcx>>,
    ) -> bool {
        let tcx = infcx.tcx;

        let TypeTest { generic_kind, lower_bound, span: _, verify_bound: _ } = type_test;

        let generic_ty = generic_kind.to_ty(tcx);
        let Some(subject) = self.try_promote_type_test_subject(infcx, generic_ty) else {
            return false;
        };

        debug!("subject = {:?}", subject);

        let r_scc = self.constraint_sccs.scc(*lower_bound);

        debug!(
            "lower_bound = {:?} r_scc={:?} universe={:?}",
            lower_bound, r_scc, self.scc_universes[r_scc]
        );

        // If the type test requires that `T: 'a` where `'a` is a
        // placeholder from another universe, that effectively requires
        // `T: 'static`, so we have to propagate that requirement.
        //
        // It doesn't matter *what* universe because the promoted `T` will
        // always be in the root universe.
        if let Some(p) = self.scc_values.placeholders_contained_in(r_scc).next() {
            debug!("encountered placeholder in higher universe: {:?}, requiring 'static", p);
            let static_r = self.universal_regions.fr_static;
            propagated_outlives_requirements.push(ClosureOutlivesRequirement {
                subject,
                outlived_free_region: static_r,
                blame_span: type_test.span,
                category: ConstraintCategory::Boring,
            });

            // we can return here -- the code below might push add'l constraints
            // but they would all be weaker than this one.
            return true;
        }

        // For each region outlived by lower_bound find a non-local,
        // universal region (it may be the same region) and add it to
        // `ClosureOutlivesRequirement`.
        for ur in self.scc_values.universal_regions_outlived_by(r_scc) {
            debug!("universal_region_outlived_by ur={:?}", ur);
            // Check whether we can already prove that the "subject" outlives `ur`.
            // If so, we don't have to propagate this requirement to our caller.
            //
            // To continue the example from the function, if we are trying to promote
            // a requirement that `T: 'X`, and we know that `'X = '1 + '2` (i.e., the union
            // `'1` and `'2`), then in this loop `ur` will be `'1` (and `'2`). So here
            // we check whether `T: '1` is something we *can* prove. If so, no need
            // to propagate that requirement.
            //
            // This is needed because -- particularly in the case
            // where `ur` is a local bound -- we are sometimes in a
            // position to prove things that our caller cannot. See
            // #53570 for an example.
            if self.eval_verify_bound(infcx, param_env, generic_ty, ur, &type_test.verify_bound) {
                continue;
            }

            let non_local_ub = self.universal_region_relations.non_local_upper_bounds(ur);
            debug!("try_promote_type_test: non_local_ub={:?}", non_local_ub);

            // This is slightly too conservative. To show T: '1, given `'2: '1`
            // and `'3: '1` we only need to prove that T: '2 *or* T: '3, but to
            // avoid potential non-determinism we approximate this by requiring
            // T: '1 and T: '2.
            for upper_bound in non_local_ub {
                debug_assert!(self.universal_regions.is_universal_region(upper_bound));
                debug_assert!(!self.universal_regions.is_local_free_region(upper_bound));

                let requirement = ClosureOutlivesRequirement {
                    subject,
                    outlived_free_region: upper_bound,
                    blame_span: type_test.span,
                    category: ConstraintCategory::Boring,
                };
                debug!("try_promote_type_test: pushing {:#?}", requirement);
                propagated_outlives_requirements.push(requirement);
            }
        }
        true
    }

    /// When we promote a type test `T: 'r`, we have to replace all region
    /// variables in the type `T` with an equal universal region from the
    /// closure signature.
    /// This is not always possible, so this is a fallible process.
    #[instrument(level = "debug", skip(self, infcx))]
    fn try_promote_type_test_subject(
        &self,
        infcx: &InferCtxt<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<ClosureOutlivesSubject<'tcx>> {
        let tcx = infcx.tcx;

        // Opaque types' substs may include useless lifetimes.
        // We will replace them with ReStatic.
        struct OpaqueFolder<'tcx> {
            tcx: TyCtxt<'tcx>,
        }
        impl<'tcx> ty::TypeFolder<TyCtxt<'tcx>> for OpaqueFolder<'tcx> {
            fn interner(&self) -> TyCtxt<'tcx> {
                self.tcx
            }
            fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
                use ty::TypeSuperFoldable as _;
                let tcx = self.tcx;
                let &ty::Alias(ty::Opaque, ty::AliasTy { substs, def_id, .. }) = t.kind() else {
                    return t.super_fold_with(self);
                };
                let substs =
                    std::iter::zip(substs, tcx.variances_of(def_id)).map(|(arg, v)| {
                        match (arg.unpack(), v) {
                            (ty::GenericArgKind::Lifetime(_), ty::Bivariant) => {
                                tcx.lifetimes.re_static.into()
                            }
                            _ => arg.fold_with(self),
                        }
                    });
                tcx.mk_opaque(def_id, tcx.mk_substs_from_iter(substs))
            }
        }

        let ty = ty.fold_with(&mut OpaqueFolder { tcx });

        let ty = tcx.fold_regions(ty, |r, _depth| {
            let r_vid = self.to_region_vid(r);
            let r_scc = self.constraint_sccs.scc(r_vid);

            // The challenge is this. We have some region variable `r`
            // whose value is a set of CFG points and universal
            // regions. We want to find if that set is *equivalent* to
            // any of the named regions found in the closure.
            // To do so, we simply check every candidate `u_r` for equality.
            self.scc_values
                .universal_regions_outlived_by(r_scc)
                .filter(|&u_r| !self.universal_regions.is_local_free_region(u_r))
                .find(|&u_r| self.eval_equal(u_r, r_vid))
                .map(|u_r| ty::Region::new_var(tcx, u_r))
                // In the case of a failure, use `ReErased`. We will eventually
                // return `None` in this case.
                .unwrap_or(tcx.lifetimes.re_erased)
        });

        debug!("try_promote_type_test_subject: folded ty = {:?}", ty);

        // This will be true if we failed to promote some region.
        if ty.has_erased_regions() {
            return None;
        }

        Some(ClosureOutlivesSubject::Ty(ClosureOutlivesSubjectTy::bind(tcx, ty)))
    }

    /// Returns a universally quantified region that outlives the
    /// value of `r` (`r` may be existentially or universally
    /// quantified).
    ///
    /// Since `r` is (potentially) an existential region, it has some
    /// value which may include (a) any number of points in the CFG
    /// and (b) any number of `end('x)` elements of universally
    /// quantified regions. To convert this into a single universal
    /// region we do as follows:
    ///
    /// - Ignore the CFG points in `'r`. All universally quantified regions
    ///   include the CFG anyhow.
    /// - For each `end('x)` element in `'r`, compute the mutual LUB, yielding
    ///   a result `'y`.
    #[instrument(skip(self), level = "debug", ret)]
    pub(crate) fn universal_upper_bound(&self, r: RegionVid) -> RegionVid {
        debug!(r = %self.region_value_str(r));

        // Find the smallest universal region that contains all other
        // universal regions within `region`.
        let mut lub = self.universal_regions.fr_fn_body;
        let r_scc = self.constraint_sccs.scc(r);
        for ur in self.scc_values.universal_regions_outlived_by(r_scc) {
            lub = self.universal_region_relations.postdom_upper_bound(lub, ur);
        }

        lub
    }

    /// Like `universal_upper_bound`, but returns an approximation more suitable
    /// for diagnostics. If `r` contains multiple disjoint universal regions
    /// (e.g. 'a and 'b in `fn foo<'a, 'b> { ... }`, we pick the lower-numbered region.
    /// This corresponds to picking named regions over unnamed regions
    /// (e.g. picking early-bound regions over a closure late-bound region).
    ///
    /// This means that the returned value may not be a true upper bound, since
    /// only 'static is known to outlive disjoint universal regions.
    /// Therefore, this method should only be used in diagnostic code,
    /// where displaying *some* named universal region is better than
    /// falling back to 'static.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn approx_universal_upper_bound(&self, r: RegionVid) -> RegionVid {
        debug!("{}", self.region_value_str(r));

        // Find the smallest universal region that contains all other
        // universal regions within `region`.
        let mut lub = self.universal_regions.fr_fn_body;
        let r_scc = self.constraint_sccs.scc(r);
        let static_r = self.universal_regions.fr_static;
        for ur in self.scc_values.universal_regions_outlived_by(r_scc) {
            let new_lub = self.universal_region_relations.postdom_upper_bound(lub, ur);
            debug!(?ur, ?lub, ?new_lub);
            // The upper bound of two non-static regions is static: this
            // means we know nothing about the relationship between these
            // two regions. Pick a 'better' one to use when constructing
            // a diagnostic
            if ur != static_r && lub != static_r && new_lub == static_r {
                // Prefer the region with an `external_name` - this
                // indicates that the region is early-bound, so working with
                // it can produce a nicer error.
                if self.region_definition(ur).external_name.is_some() {
                    lub = ur;
                } else if self.region_definition(lub).external_name.is_some() {
                    // Leave lub unchanged
                } else {
                    // If we get here, we don't have any reason to prefer
                    // one region over the other. Just pick the
                    // one with the lower index for now.
                    lub = std::cmp::min(ur, lub);
                }
            } else {
                lub = new_lub;
            }
        }

        debug!(?r, ?lub);

        lub
    }

    /// Tests if `test` is true when applied to `lower_bound` at
    /// `point`.
    fn eval_verify_bound(
        &self,
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        generic_ty: Ty<'tcx>,
        lower_bound: RegionVid,
        verify_bound: &VerifyBound<'tcx>,
    ) -> bool {
        debug!("eval_verify_bound(lower_bound={:?}, verify_bound={:?})", lower_bound, verify_bound);

        match verify_bound {
            VerifyBound::IfEq(verify_if_eq_b) => {
                self.eval_if_eq(infcx, param_env, generic_ty, lower_bound, *verify_if_eq_b)
            }

            VerifyBound::IsEmpty => {
                let lower_bound_scc = self.constraint_sccs.scc(lower_bound);
                self.scc_values.elements_contained_in(lower_bound_scc).next().is_none()
            }

            VerifyBound::OutlivedBy(r) => {
                let r_vid = self.to_region_vid(*r);
                self.eval_outlives(r_vid, lower_bound)
            }

            VerifyBound::AnyBound(verify_bounds) => verify_bounds.iter().any(|verify_bound| {
                self.eval_verify_bound(infcx, param_env, generic_ty, lower_bound, verify_bound)
            }),

            VerifyBound::AllBounds(verify_bounds) => verify_bounds.iter().all(|verify_bound| {
                self.eval_verify_bound(infcx, param_env, generic_ty, lower_bound, verify_bound)
            }),
        }
    }

    fn eval_if_eq(
        &self,
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        generic_ty: Ty<'tcx>,
        lower_bound: RegionVid,
        verify_if_eq_b: ty::Binder<'tcx, VerifyIfEq<'tcx>>,
    ) -> bool {
        let generic_ty = self.normalize_to_scc_representatives(infcx.tcx, generic_ty);
        let verify_if_eq_b = self.normalize_to_scc_representatives(infcx.tcx, verify_if_eq_b);
        match test_type_match::extract_verify_if_eq(
            infcx.tcx,
            param_env,
            &verify_if_eq_b,
            generic_ty,
        ) {
            Some(r) => {
                let r_vid = self.to_region_vid(r);
                self.eval_outlives(r_vid, lower_bound)
            }
            None => false,
        }
    }

    /// This is a conservative normalization procedure. It takes every
    /// free region in `value` and replaces it with the
    /// "representative" of its SCC (see `scc_representatives` field).
    /// We are guaranteed that if two values normalize to the same
    /// thing, then they are equal; this is a conservative check in
    /// that they could still be equal even if they normalize to
    /// different results. (For example, there might be two regions
    /// with the same value that are not in the same SCC).
    ///
    /// N.B., this is not an ideal approach and I would like to revisit
    /// it. However, it works pretty well in practice. In particular,
    /// this is needed to deal with projection outlives bounds like
    ///
    /// ```text
    /// <T as Foo<'0>>::Item: '1
    /// ```
    ///
    /// In particular, this routine winds up being important when
    /// there are bounds like `where <T as Foo<'a>>::Item: 'b` in the
    /// environment. In this case, if we can show that `'0 == 'a`,
    /// and that `'b: '1`, then we know that the clause is
    /// satisfied. In such cases, particularly due to limitations of
    /// the trait solver =), we usually wind up with a where-clause like
    /// `T: Foo<'a>` in scope, which thus forces `'0 == 'a` to be added as
    /// a constraint, and thus ensures that they are in the same SCC.
    ///
    /// So why can't we do a more correct routine? Well, we could
    /// *almost* use the `relate_tys` code, but the way it is
    /// currently setup it creates inference variables to deal with
    /// higher-ranked things and so forth, and right now the inference
    /// context is not permitted to make more inference variables. So
    /// we use this kind of hacky solution.
    fn normalize_to_scc_representatives<T>(&self, tcx: TyCtxt<'tcx>, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        tcx.fold_regions(value, |r, _db| {
            let vid = self.to_region_vid(r);
            let scc = self.constraint_sccs.scc(vid);
            let repr = self.scc_representatives[scc];
            ty::Region::new_var(tcx, repr)
        })
    }

    // Evaluate whether `sup_region == sub_region`.
    fn eval_equal(&self, r1: RegionVid, r2: RegionVid) -> bool {
        self.eval_outlives(r1, r2) && self.eval_outlives(r2, r1)
    }

    // Evaluate whether `sup_region: sub_region`.
    #[instrument(skip(self), level = "debug", ret)]
    fn eval_outlives(&self, sup_region: RegionVid, sub_region: RegionVid) -> bool {
        debug!(
            "sup_region's value = {:?} universal={:?}",
            self.region_value_str(sup_region),
            self.universal_regions.is_universal_region(sup_region),
        );
        debug!(
            "sub_region's value = {:?} universal={:?}",
            self.region_value_str(sub_region),
            self.universal_regions.is_universal_region(sub_region),
        );

        let sub_region_scc = self.constraint_sccs.scc(sub_region);
        let sup_region_scc = self.constraint_sccs.scc(sup_region);

        // If we are checking that `'sup: 'sub`, and `'sub` contains
        // some placeholder that `'sup` cannot name, then this is only
        // true if `'sup` outlives static.
        if !self.universe_compatible(sub_region_scc, sup_region_scc) {
            debug!(
                "sub universe `{sub_region_scc:?}` is not nameable \
                by super `{sup_region_scc:?}`, promoting to static",
            );

            return self.eval_outlives(sup_region, self.universal_regions.fr_static);
        }

        // Both the `sub_region` and `sup_region` consist of the union
        // of some number of universal regions (along with the union
        // of various points in the CFG; ignore those points for
        // now). Therefore, the sup-region outlives the sub-region if,
        // for each universal region R1 in the sub-region, there
        // exists some region R2 in the sup-region that outlives R1.
        let universal_outlives =
            self.scc_values.universal_regions_outlived_by(sub_region_scc).all(|r1| {
                self.scc_values
                    .universal_regions_outlived_by(sup_region_scc)
                    .any(|r2| self.universal_region_relations.outlives(r2, r1))
            });

        if !universal_outlives {
            debug!("sub region contains a universal region not present in super");
            return false;
        }

        // Now we have to compare all the points in the sub region and make
        // sure they exist in the sup region.

        if self.universal_regions.is_universal_region(sup_region) {
            // Micro-opt: universal regions contain all points.
            debug!("super is universal and hence contains all points");
            return true;
        }

        debug!("comparison between points in sup/sub");

        self.scc_values.contains_points(sup_region_scc, sub_region_scc)
    }

    /// Once regions have been propagated, this method is used to see
    /// whether any of the constraints were too strong. In particular,
    /// we want to check for a case where a universally quantified
    /// region exceeded its bounds. Consider:
    /// ```compile_fail
    /// fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    /// ```
    /// In this case, returning `x` requires `&'a u32 <: &'b u32`
    /// and hence we establish (transitively) a constraint that
    /// `'a: 'b`. The `propagate_constraints` code above will
    /// therefore add `end('a)` into the region for `'b` -- but we
    /// have no evidence that `'b` outlives `'a`, so we want to report
    /// an error.
    ///
    /// If `propagated_outlives_requirements` is `Some`, then we will
    /// push unsatisfied obligations into there. Otherwise, we'll
    /// report them as errors.
    fn check_universal_regions(
        &self,
        mut propagated_outlives_requirements: Option<&mut Vec<ClosureOutlivesRequirement<'tcx>>>,
        errors_buffer: &mut RegionErrors<'tcx>,
    ) {
        for (fr, fr_definition) in self.definitions.iter_enumerated() {
            match fr_definition.origin {
                NllRegionVariableOrigin::FreeRegion => {
                    // Go through each of the universal regions `fr` and check that
                    // they did not grow too large, accumulating any requirements
                    // for our caller into the `outlives_requirements` vector.
                    self.check_universal_region(
                        fr,
                        &mut propagated_outlives_requirements,
                        errors_buffer,
                    );
                }

                NllRegionVariableOrigin::Placeholder(placeholder) => {
                    self.check_bound_universal_region(fr, placeholder, errors_buffer);
                }

                NllRegionVariableOrigin::Existential { .. } => {
                    // nothing to check here
                }
            }
        }
    }

    /// Checks if Polonius has found any unexpected free region relations.
    ///
    /// In Polonius terms, a "subset error" (or "illegal subset relation error") is the equivalent
    /// of NLL's "checking if any region constraints were too strong": a placeholder origin `'a`
    /// was unexpectedly found to be a subset of another placeholder origin `'b`, and means in NLL
    /// terms that the "longer free region" `'a` outlived the "shorter free region" `'b`.
    ///
    /// More details can be found in this blog post by Niko:
    /// <https://smallcultfollowing.com/babysteps/blog/2019/01/17/polonius-and-region-errors/>
    ///
    /// In the canonical example
    /// ```compile_fail
    /// fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    /// ```
    /// returning `x` requires `&'a u32 <: &'b u32` and hence we establish (transitively) a
    /// constraint that `'a: 'b`. It is an error that we have no evidence that this
    /// constraint holds.
    ///
    /// If `propagated_outlives_requirements` is `Some`, then we will
    /// push unsatisfied obligations into there. Otherwise, we'll
    /// report them as errors.
    fn check_polonius_subset_errors(
        &self,
        mut propagated_outlives_requirements: Option<&mut Vec<ClosureOutlivesRequirement<'tcx>>>,
        errors_buffer: &mut RegionErrors<'tcx>,
        polonius_output: Rc<PoloniusOutput>,
    ) {
        debug!(
            "check_polonius_subset_errors: {} subset_errors",
            polonius_output.subset_errors.len()
        );

        // Similarly to `check_universal_regions`: a free region relation, which was not explicitly
        // declared ("known") was found by Polonius, so emit an error, or propagate the
        // requirements for our caller into the `propagated_outlives_requirements` vector.
        //
        // Polonius doesn't model regions ("origins") as CFG-subsets or durations, but the
        // `longer_fr` and `shorter_fr` terminology will still be used here, for consistency with
        // the rest of the NLL infrastructure. The "subset origin" is the "longer free region",
        // and the "superset origin" is the outlived "shorter free region".
        //
        // Note: Polonius will produce a subset error at every point where the unexpected
        // `longer_fr`'s "placeholder loan" is contained in the `shorter_fr`. This can be helpful
        // for diagnostics in the future, e.g. to point more precisely at the key locations
        // requiring this constraint to hold. However, the error and diagnostics code downstream
        // expects that these errors are not duplicated (and that they are in a certain order).
        // Otherwise, diagnostics messages such as the ones giving names like `'1` to elided or
        // anonymous lifetimes for example, could give these names differently, while others like
        // the outlives suggestions or the debug output from `#[rustc_regions]` would be
        // duplicated. The polonius subset errors are deduplicated here, while keeping the
        // CFG-location ordering.
        // We can iterate the HashMap here because the result is sorted afterwards.
        #[allow(rustc::potential_query_instability)]
        let mut subset_errors: Vec<_> = polonius_output
            .subset_errors
            .iter()
            .flat_map(|(_location, subset_errors)| subset_errors.iter())
            .collect();
        subset_errors.sort();
        subset_errors.dedup();

        for (longer_fr, shorter_fr) in subset_errors.into_iter() {
            debug!(
                "check_polonius_subset_errors: subset_error longer_fr={:?},\
                 shorter_fr={:?}",
                longer_fr, shorter_fr
            );

            let propagated = self.try_propagate_universal_region_error(
                *longer_fr,
                *shorter_fr,
                &mut propagated_outlives_requirements,
            );
            if propagated == RegionRelationCheckResult::Error {
                errors_buffer.push(RegionErrorKind::RegionError {
                    longer_fr: *longer_fr,
                    shorter_fr: *shorter_fr,
                    fr_origin: NllRegionVariableOrigin::FreeRegion,
                    is_reported: true,
                });
            }
        }

        // Handle the placeholder errors as usual, until the chalk-rustc-polonius triumvirate has
        // a more complete picture on how to separate this responsibility.
        for (fr, fr_definition) in self.definitions.iter_enumerated() {
            match fr_definition.origin {
                NllRegionVariableOrigin::FreeRegion => {
                    // handled by polonius above
                }

                NllRegionVariableOrigin::Placeholder(placeholder) => {
                    self.check_bound_universal_region(fr, placeholder, errors_buffer);
                }

                NllRegionVariableOrigin::Existential { .. } => {
                    // nothing to check here
                }
            }
        }
    }

    /// Checks the final value for the free region `fr` to see if it
    /// grew too large. In particular, examine what `end(X)` points
    /// wound up in `fr`'s final value; for each `end(X)` where `X !=
    /// fr`, we want to check that `fr: X`. If not, that's either an
    /// error, or something we have to propagate to our creator.
    ///
    /// Things that are to be propagated are accumulated into the
    /// `outlives_requirements` vector.
    #[instrument(skip(self, propagated_outlives_requirements, errors_buffer), level = "debug")]
    fn check_universal_region(
        &self,
        longer_fr: RegionVid,
        propagated_outlives_requirements: &mut Option<&mut Vec<ClosureOutlivesRequirement<'tcx>>>,
        errors_buffer: &mut RegionErrors<'tcx>,
    ) {
        let longer_fr_scc = self.constraint_sccs.scc(longer_fr);

        // Because this free region must be in the ROOT universe, we
        // know it cannot contain any bound universes.
        assert!(self.scc_universes[longer_fr_scc] == ty::UniverseIndex::ROOT);
        debug_assert!(self.scc_values.placeholders_contained_in(longer_fr_scc).next().is_none());

        // Only check all of the relations for the main representative of each
        // SCC, otherwise just check that we outlive said representative. This
        // reduces the number of redundant relations propagated out of
        // closures.
        // Note that the representative will be a universal region if there is
        // one in this SCC, so we will always check the representative here.
        let representative = self.scc_representatives[longer_fr_scc];
        if representative != longer_fr {
            if let RegionRelationCheckResult::Error = self.check_universal_region_relation(
                longer_fr,
                representative,
                propagated_outlives_requirements,
            ) {
                errors_buffer.push(RegionErrorKind::RegionError {
                    longer_fr,
                    shorter_fr: representative,
                    fr_origin: NllRegionVariableOrigin::FreeRegion,
                    is_reported: true,
                });
            }
            return;
        }

        // Find every region `o` such that `fr: o`
        // (because `fr` includes `end(o)`).
        let mut error_reported = false;
        for shorter_fr in self.scc_values.universal_regions_outlived_by(longer_fr_scc) {
            if let RegionRelationCheckResult::Error = self.check_universal_region_relation(
                longer_fr,
                shorter_fr,
                propagated_outlives_requirements,
            ) {
                // We only report the first region error. Subsequent errors are hidden so as
                // not to overwhelm the user, but we do record them so as to potentially print
                // better diagnostics elsewhere...
                errors_buffer.push(RegionErrorKind::RegionError {
                    longer_fr,
                    shorter_fr,
                    fr_origin: NllRegionVariableOrigin::FreeRegion,
                    is_reported: !error_reported,
                });

                error_reported = true;
            }
        }
    }

    /// Checks that we can prove that `longer_fr: shorter_fr`. If we can't we attempt to propagate
    /// the constraint outward (e.g. to a closure environment), but if that fails, there is an
    /// error.
    fn check_universal_region_relation(
        &self,
        longer_fr: RegionVid,
        shorter_fr: RegionVid,
        propagated_outlives_requirements: &mut Option<&mut Vec<ClosureOutlivesRequirement<'tcx>>>,
    ) -> RegionRelationCheckResult {
        // If it is known that `fr: o`, carry on.
        if self.universal_region_relations.outlives(longer_fr, shorter_fr) {
            RegionRelationCheckResult::Ok
        } else {
            // If we are not in a context where we can't propagate errors, or we
            // could not shrink `fr` to something smaller, then just report an
            // error.
            //
            // Note: in this case, we use the unapproximated regions to report the
            // error. This gives better error messages in some cases.
            self.try_propagate_universal_region_error(
                longer_fr,
                shorter_fr,
                propagated_outlives_requirements,
            )
        }
    }

    /// Attempt to propagate a region error (e.g. `'a: 'b`) that is not met to a closure's
    /// creator. If we cannot, then the caller should report an error to the user.
    fn try_propagate_universal_region_error(
        &self,
        longer_fr: RegionVid,
        shorter_fr: RegionVid,
        propagated_outlives_requirements: &mut Option<&mut Vec<ClosureOutlivesRequirement<'tcx>>>,
    ) -> RegionRelationCheckResult {
        if let Some(propagated_outlives_requirements) = propagated_outlives_requirements {
            // Shrink `longer_fr` until we find a non-local region (if we do).
            // We'll call it `fr-` -- it's ever so slightly smaller than
            // `longer_fr`.
            if let Some(fr_minus) = self.universal_region_relations.non_local_lower_bound(longer_fr)
            {
                debug!("try_propagate_universal_region_error: fr_minus={:?}", fr_minus);

                let blame_span_category = self.find_outlives_blame_span(
                    longer_fr,
                    NllRegionVariableOrigin::FreeRegion,
                    shorter_fr,
                );

                // Grow `shorter_fr` until we find some non-local regions. (We
                // always will.)  We'll call them `shorter_fr+` -- they're ever
                // so slightly larger than `shorter_fr`.
                let shorter_fr_plus =
                    self.universal_region_relations.non_local_upper_bounds(shorter_fr);
                debug!(
                    "try_propagate_universal_region_error: shorter_fr_plus={:?}",
                    shorter_fr_plus
                );
                for fr in shorter_fr_plus {
                    // Push the constraint `fr-: shorter_fr+`
                    propagated_outlives_requirements.push(ClosureOutlivesRequirement {
                        subject: ClosureOutlivesSubject::Region(fr_minus),
                        outlived_free_region: fr,
                        blame_span: blame_span_category.1.span,
                        category: blame_span_category.0,
                    });
                }
                return RegionRelationCheckResult::Propagated;
            }
        }

        RegionRelationCheckResult::Error
    }

    fn check_bound_universal_region(
        &self,
        longer_fr: RegionVid,
        placeholder: ty::PlaceholderRegion,
        errors_buffer: &mut RegionErrors<'tcx>,
    ) {
        debug!("check_bound_universal_region(fr={:?}, placeholder={:?})", longer_fr, placeholder,);

        let longer_fr_scc = self.constraint_sccs.scc(longer_fr);
        debug!("check_bound_universal_region: longer_fr_scc={:?}", longer_fr_scc,);

        for error_element in self.scc_values.elements_contained_in(longer_fr_scc) {
            match error_element {
                RegionElement::Location(_) | RegionElement::RootUniversalRegion(_) => {}
                // If we have some bound universal region `'a`, then the only
                // elements it can contain is itself -- we don't know anything
                // else about it!
                RegionElement::PlaceholderRegion(placeholder1) => {
                    if placeholder == placeholder1 {
                        continue;
                    }
                }
            }

            errors_buffer.push(RegionErrorKind::BoundUniversalRegionError {
                longer_fr,
                error_element,
                placeholder,
            });

            // Stop after the first error, it gets too noisy otherwise, and does not provide more information.
            break;
        }
        debug!("check_bound_universal_region: all bounds satisfied");
    }

    #[instrument(level = "debug", skip(self, infcx, errors_buffer))]
    fn check_member_constraints(
        &self,
        infcx: &InferCtxt<'tcx>,
        errors_buffer: &mut RegionErrors<'tcx>,
    ) {
        let member_constraints = self.member_constraints.clone();
        for m_c_i in member_constraints.all_indices() {
            debug!(?m_c_i);
            let m_c = &member_constraints[m_c_i];
            let member_region_vid = m_c.member_region_vid;
            debug!(
                ?member_region_vid,
                value = ?self.region_value_str(member_region_vid),
            );
            let choice_regions = member_constraints.choice_regions(m_c_i);
            debug!(?choice_regions);

            // Did the member region wind up equal to any of the option regions?
            if let Some(o) =
                choice_regions.iter().find(|&&o_r| self.eval_equal(o_r, m_c.member_region_vid))
            {
                debug!("evaluated as equal to {:?}", o);
                continue;
            }

            // If not, report an error.
            let member_region = ty::Region::new_var(infcx.tcx, member_region_vid);
            errors_buffer.push(RegionErrorKind::UnexpectedHiddenRegion {
                span: m_c.definition_span,
                hidden_ty: m_c.hidden_ty,
                key: m_c.key,
                member_region,
            });
        }
    }

    /// We have a constraint `fr1: fr2` that is not satisfied, where
    /// `fr2` represents some universal region. Here, `r` is some
    /// region where we know that `fr1: r` and this function has the
    /// job of determining whether `r` is "to blame" for the fact that
    /// `fr1: fr2` is required.
    ///
    /// This is true under two conditions:
    ///
    /// - `r == fr2`
    /// - `fr2` is `'static` and `r` is some placeholder in a universe
    ///   that cannot be named by `fr1`; in that case, we will require
    ///   that `fr1: 'static` because it is the only way to `fr1: r` to
    ///   be satisfied. (See `add_incompatible_universe`.)
    pub(crate) fn provides_universal_region(
        &self,
        r: RegionVid,
        fr1: RegionVid,
        fr2: RegionVid,
    ) -> bool {
        debug!("provides_universal_region(r={:?}, fr1={:?}, fr2={:?})", r, fr1, fr2);
        let result = {
            r == fr2 || {
                fr2 == self.universal_regions.fr_static && self.cannot_name_placeholder(fr1, r)
            }
        };
        debug!("provides_universal_region: result = {:?}", result);
        result
    }

    /// If `r2` represents a placeholder region, then this returns
    /// `true` if `r1` cannot name that placeholder in its
    /// value; otherwise, returns `false`.
    pub(crate) fn cannot_name_placeholder(&self, r1: RegionVid, r2: RegionVid) -> bool {
        debug!("cannot_name_value_of(r1={:?}, r2={:?})", r1, r2);

        match self.definitions[r2].origin {
            NllRegionVariableOrigin::Placeholder(placeholder) => {
                let universe1 = self.definitions[r1].universe;
                debug!(
                    "cannot_name_value_of: universe1={:?} placeholder={:?}",
                    universe1, placeholder
                );
                universe1.cannot_name(placeholder.universe)
            }

            NllRegionVariableOrigin::FreeRegion | NllRegionVariableOrigin::Existential { .. } => {
                false
            }
        }
    }

    /// Finds a good `ObligationCause` to blame for the fact that `fr1` outlives `fr2`.
    pub(crate) fn find_outlives_blame_span(
        &self,
        fr1: RegionVid,
        fr1_origin: NllRegionVariableOrigin,
        fr2: RegionVid,
    ) -> (ConstraintCategory<'tcx>, ObligationCause<'tcx>) {
        let BlameConstraint { category, cause, .. } = self
            .best_blame_constraint(fr1, fr1_origin, |r| self.provides_universal_region(r, fr1, fr2))
            .0;
        (category, cause)
    }

    /// Walks the graph of constraints (where `'a: 'b` is considered
    /// an edge `'a -> 'b`) to find all paths from `from_region` to
    /// `to_region`. The paths are accumulated into the vector
    /// `results`. The paths are stored as a series of
    /// `ConstraintIndex` values -- in other words, a list of *edges*.
    ///
    /// Returns: a series of constraints as well as the region `R`
    /// that passed the target test.
    pub(crate) fn find_constraint_paths_between_regions(
        &self,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
    ) -> Option<(Vec<OutlivesConstraint<'tcx>>, RegionVid)> {
        let mut context = IndexVec::from_elem(Trace::NotVisited, &self.definitions);
        context[from_region] = Trace::StartRegion;

        // Use a deque so that we do a breadth-first search. We will
        // stop at the first match, which ought to be the shortest
        // path (fewest constraints).
        let mut deque = VecDeque::new();
        deque.push_back(from_region);

        while let Some(r) = deque.pop_front() {
            debug!(
                "find_constraint_paths_between_regions: from_region={:?} r={:?} value={}",
                from_region,
                r,
                self.region_value_str(r),
            );

            // Check if we reached the region we were looking for. If so,
            // we can reconstruct the path that led to it and return it.
            if target_test(r) {
                let mut result = vec![];
                let mut p = r;
                loop {
                    match context[p].clone() {
                        Trace::NotVisited => {
                            bug!("found unvisited region {:?} on path to {:?}", p, r)
                        }

                        Trace::FromOutlivesConstraint(c) => {
                            p = c.sup;
                            result.push(c);
                        }

                        Trace::StartRegion => {
                            result.reverse();
                            return Some((result, r));
                        }
                    }
                }
            }

            // Otherwise, walk over the outgoing constraints and
            // enqueue any regions we find, keeping track of how we
            // reached them.

            // A constraint like `'r: 'x` can come from our constraint
            // graph.
            let fr_static = self.universal_regions.fr_static;
            let outgoing_edges_from_graph =
                self.constraint_graph.outgoing_edges(r, &self.constraints, fr_static);

            // Always inline this closure because it can be hot.
            let mut handle_constraint = #[inline(always)]
            |constraint: OutlivesConstraint<'tcx>| {
                debug_assert_eq!(constraint.sup, r);
                let sub_region = constraint.sub;
                if let Trace::NotVisited = context[sub_region] {
                    context[sub_region] = Trace::FromOutlivesConstraint(constraint);
                    deque.push_back(sub_region);
                }
            };

            // This loop can be hot.
            for constraint in outgoing_edges_from_graph {
                handle_constraint(constraint);
            }

            // Member constraints can also give rise to `'r: 'x` edges that
            // were not part of the graph initially, so watch out for those.
            // (But they are extremely rare; this loop is very cold.)
            for constraint in self.applied_member_constraints(r) {
                let p_c = &self.member_constraints[constraint.member_constraint_index];
                let constraint = OutlivesConstraint {
                    sup: r,
                    sub: constraint.min_choice,
                    locations: Locations::All(p_c.definition_span),
                    span: p_c.definition_span,
                    category: ConstraintCategory::OpaqueType,
                    variance_info: ty::VarianceDiagInfo::default(),
                    from_closure: false,
                };
                handle_constraint(constraint);
            }
        }

        None
    }

    /// Finds some region R such that `fr1: R` and `R` is live at `elem`.
    #[instrument(skip(self), level = "trace", ret)]
    pub(crate) fn find_sub_region_live_at(&self, fr1: RegionVid, elem: Location) -> RegionVid {
        trace!(scc = ?self.constraint_sccs.scc(fr1));
        trace!(universe = ?self.scc_universes[self.constraint_sccs.scc(fr1)]);
        self.find_constraint_paths_between_regions(fr1, |r| {
            // First look for some `r` such that `fr1: r` and `r` is live at `elem`
            trace!(?r, liveness_constraints=?self.liveness_constraints.region_value_str(r));
            self.liveness_constraints.contains(r, elem)
        })
        .or_else(|| {
            // If we fail to find that, we may find some `r` such that
            // `fr1: r` and `r` is a placeholder from some universe
            // `fr1` cannot name. This would force `fr1` to be
            // `'static`.
            self.find_constraint_paths_between_regions(fr1, |r| {
                self.cannot_name_placeholder(fr1, r)
            })
        })
        .or_else(|| {
            // If we fail to find THAT, it may be that `fr1` is a
            // placeholder that cannot "fit" into its SCC. In that
            // case, there should be some `r` where `fr1: r` and `fr1` is a
            // placeholder that `r` cannot name. We can blame that
            // edge.
            //
            // Remember that if `R1: R2`, then the universe of R1
            // must be able to name the universe of R2, because R2 will
            // be at least `'empty(Universe(R2))`, and `R1` must be at
            // larger than that.
            self.find_constraint_paths_between_regions(fr1, |r| {
                self.cannot_name_placeholder(r, fr1)
            })
        })
        .map(|(_path, r)| r)
        .unwrap()
    }

    /// Get the region outlived by `longer_fr` and live at `element`.
    pub(crate) fn region_from_element(
        &self,
        longer_fr: RegionVid,
        element: &RegionElement,
    ) -> RegionVid {
        match *element {
            RegionElement::Location(l) => self.find_sub_region_live_at(longer_fr, l),
            RegionElement::RootUniversalRegion(r) => r,
            RegionElement::PlaceholderRegion(error_placeholder) => self
                .definitions
                .iter_enumerated()
                .find_map(|(r, definition)| match definition.origin {
                    NllRegionVariableOrigin::Placeholder(p) if p == error_placeholder => Some(r),
                    _ => None,
                })
                .unwrap(),
        }
    }

    /// Get the region definition of `r`.
    pub(crate) fn region_definition(&self, r: RegionVid) -> &RegionDefinition<'tcx> {
        &self.definitions[r]
    }

    /// Check if the SCC of `r` contains `upper`.
    pub(crate) fn upper_bound_in_region_scc(&self, r: RegionVid, upper: RegionVid) -> bool {
        let r_scc = self.constraint_sccs.scc(r);
        self.scc_values.contains(r_scc, upper)
    }

    pub(crate) fn universal_regions(&self) -> &UniversalRegions<'tcx> {
        self.universal_regions.as_ref()
    }

    /// Tries to find the best constraint to blame for the fact that
    /// `R: from_region`, where `R` is some region that meets
    /// `target_test`. This works by following the constraint graph,
    /// creating a constraint path that forces `R` to outlive
    /// `from_region`, and then finding the best choices within that
    /// path to blame.
    #[instrument(level = "debug", skip(self, target_test))]
    pub(crate) fn best_blame_constraint(
        &self,
        from_region: RegionVid,
        from_region_origin: NllRegionVariableOrigin,
        target_test: impl Fn(RegionVid) -> bool,
    ) -> (BlameConstraint<'tcx>, Vec<ExtraConstraintInfo>) {
        // Find all paths
        let (path, target_region) =
            self.find_constraint_paths_between_regions(from_region, target_test).unwrap();
        debug!(
            "path={:#?}",
            path.iter()
                .map(|c| format!(
                    "{:?} ({:?}: {:?})",
                    c,
                    self.constraint_sccs.scc(c.sup),
                    self.constraint_sccs.scc(c.sub),
                ))
                .collect::<Vec<_>>()
        );

        let mut extra_info = vec![];
        for constraint in path.iter() {
            let outlived = constraint.sub;
            let Some(origin) = self.var_infos.get(outlived) else { continue; };
            let RegionVariableOrigin::Nll(NllRegionVariableOrigin::Placeholder(p)) = origin.origin else { continue; };
            debug!(?constraint, ?p);
            let ConstraintCategory::Predicate(span) = constraint.category else { continue; };
            extra_info.push(ExtraConstraintInfo::PlaceholderFromPredicate(span));
            // We only want to point to one
            break;
        }

        // We try to avoid reporting a `ConstraintCategory::Predicate` as our best constraint.
        // Instead, we use it to produce an improved `ObligationCauseCode`.
        // FIXME - determine what we should do if we encounter multiple `ConstraintCategory::Predicate`
        // constraints. Currently, we just pick the first one.
        let cause_code = path
            .iter()
            .find_map(|constraint| {
                if let ConstraintCategory::Predicate(predicate_span) = constraint.category {
                    // We currently do not store the `DefId` in the `ConstraintCategory`
                    // for performances reasons. The error reporting code used by NLL only
                    // uses the span, so this doesn't cause any problems at the moment.
                    Some(ObligationCauseCode::BindingObligation(
                        CRATE_DEF_ID.to_def_id(),
                        predicate_span,
                    ))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| ObligationCauseCode::MiscObligation);

        // Classify each of the constraints along the path.
        let mut categorized_path: Vec<BlameConstraint<'tcx>> = path
            .iter()
            .map(|constraint| BlameConstraint {
                category: constraint.category,
                from_closure: constraint.from_closure,
                cause: ObligationCause::new(constraint.span, CRATE_DEF_ID, cause_code.clone()),
                variance_info: constraint.variance_info,
                outlives_constraint: *constraint,
            })
            .collect();
        debug!("categorized_path={:#?}", categorized_path);

        // To find the best span to cite, we first try to look for the
        // final constraint that is interesting and where the `sup` is
        // not unified with the ultimate target region. The reason
        // for this is that we have a chain of constraints that lead
        // from the source to the target region, something like:
        //
        //    '0: '1 ('0 is the source)
        //    '1: '2
        //    '2: '3
        //    '3: '4
        //    '4: '5
        //    '5: '6 ('6 is the target)
        //
        // Some of those regions are unified with `'6` (in the same
        // SCC). We want to screen those out. After that point, the
        // "closest" constraint we have to the end is going to be the
        // most likely to be the point where the value escapes -- but
        // we still want to screen for an "interesting" point to
        // highlight (e.g., a call site or something).
        let target_scc = self.constraint_sccs.scc(target_region);
        let mut range = 0..path.len();

        // As noted above, when reporting an error, there is typically a chain of constraints
        // leading from some "source" region which must outlive some "target" region.
        // In most cases, we prefer to "blame" the constraints closer to the target --
        // but there is one exception. When constraints arise from higher-ranked subtyping,
        // we generally prefer to blame the source value,
        // as the "target" in this case tends to be some type annotation that the user gave.
        // Therefore, if we find that the region origin is some instantiation
        // of a higher-ranked region, we start our search from the "source" point
        // rather than the "target", and we also tweak a few other things.
        //
        // An example might be this bit of Rust code:
        //
        // ```rust
        // let x: fn(&'static ()) = |_| {};
        // let y: for<'a> fn(&'a ()) = x;
        // ```
        //
        // In MIR, this will be converted into a combination of assignments and type ascriptions.
        // In particular, the 'static is imposed through a type ascription:
        //
        // ```rust
        // x = ...;
        // AscribeUserType(x, fn(&'static ())
        // y = x;
        // ```
        //
        // We wind up ultimately with constraints like
        //
        // ```rust
        // !a: 'temp1 // from the `y = x` statement
        // 'temp1: 'temp2
        // 'temp2: 'static // from the AscribeUserType
        // ```
        //
        // and here we prefer to blame the source (the y = x statement).
        let blame_source = match from_region_origin {
            NllRegionVariableOrigin::FreeRegion
            | NllRegionVariableOrigin::Existential { from_forall: false } => true,
            NllRegionVariableOrigin::Placeholder(_)
            | NllRegionVariableOrigin::Existential { from_forall: true } => false,
        };

        let find_region = |i: &usize| {
            let constraint = &path[*i];

            let constraint_sup_scc = self.constraint_sccs.scc(constraint.sup);

            if blame_source {
                match categorized_path[*i].category {
                    ConstraintCategory::OpaqueType
                    | ConstraintCategory::Boring
                    | ConstraintCategory::BoringNoLocation
                    | ConstraintCategory::Internal
                    | ConstraintCategory::Predicate(_) => false,
                    ConstraintCategory::TypeAnnotation
                    | ConstraintCategory::Return(_)
                    | ConstraintCategory::Yield => true,
                    _ => constraint_sup_scc != target_scc,
                }
            } else {
                !matches!(
                    categorized_path[*i].category,
                    ConstraintCategory::OpaqueType
                        | ConstraintCategory::Boring
                        | ConstraintCategory::BoringNoLocation
                        | ConstraintCategory::Internal
                        | ConstraintCategory::Predicate(_)
                )
            }
        };

        let best_choice =
            if blame_source { range.rev().find(find_region) } else { range.find(find_region) };

        debug!(?best_choice, ?blame_source, ?extra_info);

        if let Some(i) = best_choice {
            if let Some(next) = categorized_path.get(i + 1) {
                if matches!(categorized_path[i].category, ConstraintCategory::Return(_))
                    && next.category == ConstraintCategory::OpaqueType
                {
                    // The return expression is being influenced by the return type being
                    // impl Trait, point at the return type and not the return expr.
                    return (next.clone(), extra_info);
                }
            }

            if categorized_path[i].category == ConstraintCategory::Return(ReturnConstraint::Normal)
            {
                let field = categorized_path.iter().find_map(|p| {
                    if let ConstraintCategory::ClosureUpvar(f) = p.category {
                        Some(f)
                    } else {
                        None
                    }
                });

                if let Some(field) = field {
                    categorized_path[i].category =
                        ConstraintCategory::Return(ReturnConstraint::ClosureUpvar(field));
                }
            }

            return (categorized_path[i].clone(), extra_info);
        }

        // If that search fails, that is.. unusual. Maybe everything
        // is in the same SCC or something. In that case, find what
        // appears to be the most interesting point to report to the
        // user via an even more ad-hoc guess.
        categorized_path.sort_by_key(|p| p.category);
        debug!("sorted_path={:#?}", categorized_path);

        (categorized_path.remove(0), extra_info)
    }

    pub(crate) fn universe_info(&self, universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        self.universe_causes[&universe].clone()
    }

    /// Tries to find the terminator of the loop in which the region 'r' resides.
    /// Returns the location of the terminator if found.
    pub(crate) fn find_loop_terminator_location(
        &self,
        r: RegionVid,
        body: &Body<'_>,
    ) -> Option<Location> {
        let scc = self.constraint_sccs.scc(r);
        let locations = self.scc_values.locations_outlived_by(scc);
        for location in locations {
            let bb = &body[location.block];
            if let Some(terminator) = &bb.terminator {
                // terminator of a loop should be TerminatorKind::FalseUnwind
                if let TerminatorKind::FalseUnwind { .. } = terminator.kind {
                    return Some(location);
                }
            }
        }
        None
    }
}

impl<'tcx> RegionDefinition<'tcx> {
    fn new(universe: ty::UniverseIndex, rv_origin: RegionVariableOrigin) -> Self {
        // Create a new region definition. Note that, for free
        // regions, the `external_name` field gets updated later in
        // `init_universal_regions`.

        let origin = match rv_origin {
            RegionVariableOrigin::Nll(origin) => origin,
            _ => NllRegionVariableOrigin::Existential { from_forall: false },
        };

        Self { origin, universe, external_name: None }
    }
}

#[derive(Clone, Debug)]
pub struct BlameConstraint<'tcx> {
    pub category: ConstraintCategory<'tcx>,
    pub from_closure: bool,
    pub cause: ObligationCause<'tcx>,
    pub variance_info: ty::VarianceDiagInfo<'tcx>,
    pub outlives_constraint: OutlivesConstraint<'tcx>,
}
