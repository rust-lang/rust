use std::collections::VecDeque;
use std::rc::Rc;

use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::scc::Sccs;
use rustc_errors::Diag;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_index::IndexVec;
use rustc_infer::infer::outlives::test_type_match;
use rustc_infer::infer::region_constraints::{GenericKind, VerifyBound, VerifyIfEq};
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin};
use rustc_middle::bug;
use rustc_middle::mir::{
    AnnotationSource, BasicBlock, Body, ConstraintCategory, Local, Location, ReturnConstraint,
    TerminatorKind,
};
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt, TypeFoldable, UniverseIndex, fold_regions};
use rustc_mir_dataflow::points::DenseLocationMap;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::{DUMMY_SP, Span};
use tracing::{Level, debug, enabled, instrument, trace};

use crate::constraints::graph::NormalConstraintGraph;
use crate::constraints::{ConstraintSccIndex, OutlivesConstraint, OutlivesConstraintSet};
use crate::dataflow::BorrowIndex;
use crate::diagnostics::{RegionErrorKind, RegionErrors, UniverseInfo};
use crate::handle_placeholders::{LoweredConstraints, RegionTracker};
use crate::polonius::LiveLoans;
use crate::polonius::legacy::PoloniusOutput;
use crate::region_infer::values::{LivenessValues, RegionElement, RegionValues, ToElementIndex};
use crate::type_check::Locations;
use crate::type_check::free_region_relations::UniversalRegionRelations;
use crate::universal_regions::UniversalRegions;
use crate::{
    BorrowckInferCtxt, ClosureOutlivesRequirement, ClosureOutlivesSubject,
    ClosureOutlivesSubjectTy, ClosureRegionRequirements,
};

mod dump_mir;
mod graphviz;
pub(crate) mod opaque_types;
mod reverse_sccs;

pub(crate) mod values;

/// The representative region variable for an SCC, tagged by its origin.
/// We prefer placeholders over existentially quantified variables, otherwise
/// it's the one with the smallest Region Variable ID. In other words,
/// the order of this enumeration really matters!
#[derive(Copy, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub(crate) enum Representative {
    FreeRegion(RegionVid),
    Placeholder(RegionVid),
    Existential(RegionVid),
}

impl Representative {
    pub(crate) fn rvid(self) -> RegionVid {
        match self {
            Representative::FreeRegion(region_vid)
            | Representative::Placeholder(region_vid)
            | Representative::Existential(region_vid) => region_vid,
        }
    }

    pub(crate) fn new(r: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        match definition.origin {
            NllRegionVariableOrigin::FreeRegion => Representative::FreeRegion(r),
            NllRegionVariableOrigin::Placeholder(_) => Representative::Placeholder(r),
            NllRegionVariableOrigin::Existential { .. } => Representative::Existential(r),
        }
    }
}

pub(crate) type ConstraintSccs = Sccs<RegionVid, ConstraintSccIndex>;

pub struct RegionInferenceContext<'tcx> {
    /// Contains the definition for every region variable. Region
    /// variables are identified by their index (`RegionVid`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    pub(crate) definitions: Frozen<IndexVec<RegionVid, RegionDefinition<'tcx>>>,

    /// The liveness constraints added to each region. For most
    /// regions, these start out empty and steadily grow, though for
    /// each universally quantified region R they start out containing
    /// the entire CFG and `end(R)`.
    liveness_constraints: LivenessValues,

    /// The outlives constraints computed by the type-check.
    constraints: Frozen<OutlivesConstraintSet<'tcx>>,

    /// The constraint-set, but in graph form, making it easy to traverse
    /// the constraints adjacent to a particular region. Used to construct
    /// the SCC (see `constraint_sccs`) and for error reporting.
    constraint_graph: Frozen<NormalConstraintGraph>,

    /// The SCC computed from `constraints` and the constraint
    /// graph. We have an edge from SCC A to SCC B if `A: B`. Used to
    /// compute the values of each region.
    constraint_sccs: ConstraintSccs,

    scc_annotations: IndexVec<ConstraintSccIndex, RegionTracker>,

    /// Map universe indexes to information on why we created it.
    universe_causes: FxIndexMap<ty::UniverseIndex, UniverseInfo<'tcx>>,

    /// The final inferred values of the region variables; we compute
    /// one value per SCC. To get the value for any given *region*,
    /// you first find which scc it is a part of.
    scc_values: RegionValues<ConstraintSccIndex>,

    /// Type constraints that we check after solving.
    type_tests: Vec<TypeTest<'tcx>>,

    /// Information about how the universally quantified regions in
    /// scope on this function relate to one another.
    universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
}

#[derive(Debug)]
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
pub(crate) struct TypeTest<'tcx> {
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
enum Trace<'a, 'tcx> {
    StartRegion,
    FromGraph(&'a OutlivesConstraint<'tcx>),
    FromStatic(RegionVid),
    NotVisited,
}

#[instrument(skip(infcx, sccs), level = "debug")]
fn sccs_info<'tcx>(infcx: &BorrowckInferCtxt<'tcx>, sccs: &ConstraintSccs) {
    use crate::renumber::RegionCtxt;

    let var_to_origin = infcx.reg_var_to_origin.borrow();

    let mut var_to_origin_sorted = var_to_origin.clone().into_iter().collect::<Vec<_>>();
    var_to_origin_sorted.sort_by_key(|vto| vto.0);

    if enabled!(Level::DEBUG) {
        let mut reg_vars_to_origins_str = "region variables to origins:\n".to_string();
        for (reg_var, origin) in var_to_origin_sorted.into_iter() {
            reg_vars_to_origins_str.push_str(&format!("{reg_var:?}: {origin:?}\n"));
        }
        debug!("{}", reg_vars_to_origins_str);
    }

    let num_components = sccs.num_sccs();
    let mut components = vec![FxIndexSet::default(); num_components];

    for (reg_var, scc_idx) in sccs.scc_indices().iter_enumerated() {
        let origin = var_to_origin.get(&reg_var).unwrap_or(&RegionCtxt::Unknown);
        components[scc_idx.as_usize()].insert((reg_var, *origin));
    }

    if enabled!(Level::DEBUG) {
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
    }

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
        let edge_representatives = sccs
            .successors(*scc_idx)
            .iter()
            .map(|scc_idx| components_representatives[scc_idx])
            .collect::<Vec<_>>();
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
    pub(crate) fn new(
        infcx: &BorrowckInferCtxt<'tcx>,
        lowered_constraints: LoweredConstraints<'tcx>,
        universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
        location_map: Rc<DenseLocationMap>,
    ) -> Self {
        let universal_regions = &universal_region_relations.universal_regions;

        let LoweredConstraints {
            constraint_sccs,
            definitions,
            outlives_constraints,
            scc_annotations,
            type_tests,
            liveness_constraints,
            universe_causes,
            placeholder_indices,
        } = lowered_constraints;

        debug!("universal_regions: {:#?}", universal_region_relations.universal_regions);
        debug!("outlives constraints: {:#?}", outlives_constraints);
        debug!("placeholder_indices: {:#?}", placeholder_indices);
        debug!("type tests: {:#?}", type_tests);

        let constraint_graph = Frozen::freeze(outlives_constraints.graph(definitions.len()));

        if cfg!(debug_assertions) {
            sccs_info(infcx, &constraint_sccs);
        }

        let mut scc_values =
            RegionValues::new(location_map, universal_regions.len(), placeholder_indices);

        for region in liveness_constraints.regions() {
            let scc = constraint_sccs.scc(region);
            scc_values.merge_liveness(scc, region, &liveness_constraints);
        }

        let mut result = Self {
            definitions,
            liveness_constraints,
            constraints: outlives_constraints,
            constraint_graph,
            constraint_sccs,
            scc_annotations,
            universe_causes,
            scc_values,
            type_tests,
            universal_region_relations,
        };

        result.init_free_and_bound_regions();

        result
    }

    /// Initializes the region variables for each universally
    /// quantified region (lifetime parameter). The first N variables
    /// always correspond to the regions appearing in the function
    /// signature (both named and anonymous) and where-clauses. This
    /// function iterates over those regions and initializes them with
    /// minimum values.
    ///
    /// For example:
    /// ```ignore (illustrative)
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
    ///
    /// This bit of logic also handles invalid universe relations
    /// for higher-kinded types.
    ///
    /// We Walk each SCC `A` and `B` such that `A: B`
    /// and ensure that universe(A) can see universe(B).
    ///
    /// This serves to enforce the 'empty/placeholder' hierarchy
    /// (described in more detail on `RegionKind`):
    ///
    /// ```ignore (illustrative)
    /// static -----+
    ///   |         |
    /// empty(U0) placeholder(U1)
    ///   |      /
    /// empty(U1)
    /// ```
    ///
    /// In particular, imagine we have variables R0 in U0 and R1
    /// created in U1, and constraints like this;
    ///
    /// ```ignore (illustrative)
    /// R1: !1 // R1 outlives the placeholder in U1
    /// R1: R0 // R1 outlives R0
    /// ```
    ///
    /// Here, we wish for R1 to be `'static`, because it
    /// cannot outlive `placeholder(U1)` and `empty(U0)` any other way.
    ///
    /// Thanks to this loop, what happens is that the `R1: R0`
    /// constraint has lowered the universe of `R1` to `U0`, which in turn
    /// means that the `R1: !1` constraint here will cause
    /// `R1` to become `'static`.
    fn init_free_and_bound_regions(&mut self) {
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
                    self.scc_values.add_element(scc, placeholder);
                }

                NllRegionVariableOrigin::Existential { .. } => {
                    // For existential, regions, nothing to do.
                }
            }
        }
    }

    /// Returns an iterator over all the region indices.
    pub(crate) fn regions(&self) -> impl Iterator<Item = RegionVid> + 'tcx {
        self.definitions.indices()
    }

    /// Given a universal region in scope on the MIR, returns the
    /// corresponding index.
    ///
    /// Panics if `r` is not a registered universal region, most notably
    /// if it is a placeholder. Handling placeholders requires access to the
    /// `MirTypeckRegionConstraints`.
    pub(crate) fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        self.universal_regions().to_region_vid(r)
    }

    /// Returns an iterator over all the outlives constraints.
    pub(crate) fn outlives_constraints(&self) -> impl Iterator<Item = OutlivesConstraint<'tcx>> {
        self.constraints.outlives().iter().copied()
    }

    /// Adds annotations for `#[rustc_regions]`; see `UniversalRegions::annotate`.
    pub(crate) fn annotate(&self, tcx: TyCtxt<'tcx>, err: &mut Diag<'_, ()>) {
        self.universal_regions().annotate(tcx, err)
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

    pub(crate) fn placeholders_contained_in(
        &self,
        r: RegionVid,
    ) -> impl Iterator<Item = ty::PlaceholderRegion> {
        let scc = self.constraint_sccs.scc(r);
        self.scc_values.placeholders_contained_in(scc)
    }

    /// Performs region inference and report errors if we see any
    /// unsatisfiable constraints. If this is a closure, returns the
    /// region requirements to propagate to our creator, if any.
    #[instrument(skip(self, infcx, body, polonius_output), level = "debug")]
    pub(super) fn solve(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        body: &Body<'tcx>,
        polonius_output: Option<Box<PoloniusOutput>>,
    ) -> (Option<ClosureRegionRequirements<'tcx>>, RegionErrors<'tcx>) {
        let mir_def_id = body.source.def_id();
        self.propagate_constraints();

        let mut errors_buffer = RegionErrors::new(infcx.tcx);

        // If this is a closure, we can propagate unsatisfied
        // `outlives_requirements` to our creator, so create a vector
        // to store those. Otherwise, we'll pass in `None` to the
        // functions below, which will trigger them to report errors
        // eagerly.
        let mut outlives_requirements = infcx.tcx.is_typeck_child(mir_def_id).then(Vec::new);

        self.check_type_tests(infcx, outlives_requirements.as_mut(), &mut errors_buffer);

        debug!(?errors_buffer);
        debug!(?outlives_requirements);

        // In Polonius mode, the errors about missing universal region relations are in the output
        // and need to be emitted or propagated. Otherwise, we need to check whether the
        // constraints were too strong, and if so, emit or propagate those errors.
        if infcx.tcx.sess.opts.unstable_opts.polonius.is_legacy_enabled() {
            self.check_polonius_subset_errors(
                outlives_requirements.as_mut(),
                &mut errors_buffer,
                polonius_output
                    .as_ref()
                    .expect("Polonius output is unavailable despite `-Z polonius`"),
            );
        } else {
            self.check_universal_regions(outlives_requirements.as_mut(), &mut errors_buffer);
        }

        debug!(?errors_buffer);

        let outlives_requirements = outlives_requirements.unwrap_or_default();

        if outlives_requirements.is_empty() {
            (None, errors_buffer)
        } else {
            let num_external_vids = self.universal_regions().num_global_and_external_regions();
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
    #[instrument(skip(self), level = "debug")]
    fn propagate_constraints(&mut self) {
        debug!("constraints={:#?}", {
            let mut constraints: Vec<_> = self.outlives_constraints().collect();
            constraints.sort_by_key(|c| (c.sup, c.sub));
            constraints
                .into_iter()
                .map(|c| (c, self.constraint_sccs.scc(c.sup), self.constraint_sccs.scc(c.sub)))
                .collect::<Vec<_>>()
        });

        // To propagate constraints, we walk the DAG induced by the
        // SCC. For each SCC `A`, we visit its successors and compute
        // their values, then we union all those values to get our
        // own.
        for scc_a in self.constraint_sccs.all_sccs() {
            // Walk each SCC `B` such that `A: B`...
            for &scc_b in self.constraint_sccs.successors(scc_a) {
                debug!(?scc_b);
                self.scc_values.add_region(scc_a, scc_b);
            }
        }
    }

    /// Returns `true` if all the placeholders in the value of `scc_b` are nameable
    /// in `scc_a`. Used during constraint propagation, and only once
    /// the value of `scc_b` has been computed.
    fn can_name_all_placeholders(
        &self,
        scc_a: ConstraintSccIndex,
        scc_b: ConstraintSccIndex,
    ) -> bool {
        self.scc_annotations[scc_a].can_name_all_placeholders(self.scc_annotations[scc_b])
    }

    /// Once regions have been propagated, this method is used to see
    /// whether the "type tests" produced by typeck were satisfied;
    /// type tests encode type-outlives relationships like `T:
    /// 'a`. See `TypeTest` for more details.
    fn check_type_tests(
        &self,
        infcx: &InferCtxt<'tcx>,
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
                generic_ty,
                type_test.lower_bound,
                &type_test.verify_bound,
            ) {
                continue;
            }

            if let Some(propagated_outlives_requirements) = &mut propagated_outlives_requirements
                && self.try_promote_type_test(infcx, type_test, propagated_outlives_requirements)
            {
                continue;
            }

            // Type-test failed. Report the error.
            let erased_generic_kind = infcx.tcx.erase_and_anonymize_regions(type_test.generic_kind);

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
        type_test: &TypeTest<'tcx>,
        propagated_outlives_requirements: &mut Vec<ClosureOutlivesRequirement<'tcx>>,
    ) -> bool {
        let tcx = infcx.tcx;
        let TypeTest { generic_kind, lower_bound, span: blame_span, verify_bound: _ } = *type_test;

        let generic_ty = generic_kind.to_ty(tcx);
        let Some(subject) = self.try_promote_type_test_subject(infcx, generic_ty) else {
            return false;
        };

        let r_scc = self.constraint_sccs.scc(lower_bound);
        debug!(
            "lower_bound = {:?} r_scc={:?} universe={:?}",
            lower_bound,
            r_scc,
            self.max_nameable_universe(r_scc)
        );
        // If the type test requires that `T: 'a` where `'a` is a
        // placeholder from another universe, that effectively requires
        // `T: 'static`, so we have to propagate that requirement.
        //
        // It doesn't matter *what* universe because the promoted `T` will
        // always be in the root universe.
        if let Some(p) = self.scc_values.placeholders_contained_in(r_scc).next() {
            debug!("encountered placeholder in higher universe: {:?}, requiring 'static", p);
            let static_r = self.universal_regions().fr_static;
            propagated_outlives_requirements.push(ClosureOutlivesRequirement {
                subject,
                outlived_free_region: static_r,
                blame_span,
                category: ConstraintCategory::Boring,
            });

            // we can return here -- the code below might push add'l constraints
            // but they would all be weaker than this one.
            return true;
        }

        // For each region outlived by lower_bound find a non-local,
        // universal region (it may be the same region) and add it to
        // `ClosureOutlivesRequirement`.
        let mut found_outlived_universal_region = false;
        for ur in self.scc_values.universal_regions_outlived_by(r_scc) {
            found_outlived_universal_region = true;
            debug!("universal_region_outlived_by ur={:?}", ur);
            let non_local_ub = self.universal_region_relations.non_local_upper_bounds(ur);
            debug!(?non_local_ub);

            // This is slightly too conservative. To show T: '1, given `'2: '1`
            // and `'3: '1` we only need to prove that T: '2 *or* T: '3, but to
            // avoid potential non-determinism we approximate this by requiring
            // T: '1 and T: '2.
            for upper_bound in non_local_ub {
                debug_assert!(self.universal_regions().is_universal_region(upper_bound));
                debug_assert!(!self.universal_regions().is_local_free_region(upper_bound));

                let requirement = ClosureOutlivesRequirement {
                    subject,
                    outlived_free_region: upper_bound,
                    blame_span,
                    category: ConstraintCategory::Boring,
                };
                debug!(?requirement, "adding closure requirement");
                propagated_outlives_requirements.push(requirement);
            }
        }
        // If we succeed to promote the subject, i.e. it only contains non-local regions,
        // and fail to prove the type test inside of the closure, the `lower_bound` has to
        // also be at least as large as some universal region, as the type test is otherwise
        // trivial.
        assert!(found_outlived_universal_region);
        true
    }

    /// When we promote a type test `T: 'r`, we have to replace all region
    /// variables in the type `T` with an equal universal region from the
    /// closure signature.
    /// This is not always possible, so this is a fallible process.
    #[instrument(level = "debug", skip(self, infcx), ret)]
    fn try_promote_type_test_subject(
        &self,
        infcx: &InferCtxt<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<ClosureOutlivesSubject<'tcx>> {
        let tcx = infcx.tcx;
        let mut failed = false;
        let ty = fold_regions(tcx, ty, |r, _depth| {
            let r_vid = self.to_region_vid(r);
            let r_scc = self.constraint_sccs.scc(r_vid);

            // The challenge is this. We have some region variable `r`
            // whose value is a set of CFG points and universal
            // regions. We want to find if that set is *equivalent* to
            // any of the named regions found in the closure.
            // To do so, we simply check every candidate `u_r` for equality.
            self.scc_values
                .universal_regions_outlived_by(r_scc)
                .filter(|&u_r| !self.universal_regions().is_local_free_region(u_r))
                .find(|&u_r| self.eval_equal(u_r, r_vid))
                .map(|u_r| ty::Region::new_var(tcx, u_r))
                // In case we could not find a named region to map to,
                // we will return `None` below.
                .unwrap_or_else(|| {
                    failed = true;
                    r
                })
        });

        debug!("try_promote_type_test_subject: folded ty = {:?}", ty);

        // This will be true if we failed to promote some region.
        if failed {
            return None;
        }

        Some(ClosureOutlivesSubject::Ty(ClosureOutlivesSubjectTy::bind(tcx, ty)))
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
        let mut lub = self.universal_regions().fr_fn_body;
        let r_scc = self.constraint_sccs.scc(r);
        let static_r = self.universal_regions().fr_static;
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
        generic_ty: Ty<'tcx>,
        lower_bound: RegionVid,
        verify_bound: &VerifyBound<'tcx>,
    ) -> bool {
        debug!("eval_verify_bound(lower_bound={:?}, verify_bound={:?})", lower_bound, verify_bound);

        match verify_bound {
            VerifyBound::IfEq(verify_if_eq_b) => {
                self.eval_if_eq(infcx, generic_ty, lower_bound, *verify_if_eq_b)
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
                self.eval_verify_bound(infcx, generic_ty, lower_bound, verify_bound)
            }),

            VerifyBound::AllBounds(verify_bounds) => verify_bounds.iter().all(|verify_bound| {
                self.eval_verify_bound(infcx, generic_ty, lower_bound, verify_bound)
            }),
        }
    }

    fn eval_if_eq(
        &self,
        infcx: &InferCtxt<'tcx>,
        generic_ty: Ty<'tcx>,
        lower_bound: RegionVid,
        verify_if_eq_b: ty::Binder<'tcx, VerifyIfEq<'tcx>>,
    ) -> bool {
        let generic_ty = self.normalize_to_scc_representatives(infcx.tcx, generic_ty);
        let verify_if_eq_b = self.normalize_to_scc_representatives(infcx.tcx, verify_if_eq_b);
        match test_type_match::extract_verify_if_eq(infcx.tcx, &verify_if_eq_b, generic_ty) {
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
        fold_regions(tcx, value, |r, _db| {
            let vid = self.to_region_vid(r);
            let scc = self.constraint_sccs.scc(vid);
            let repr = self.scc_representative(scc);
            ty::Region::new_var(tcx, repr)
        })
    }

    /// Evaluate whether `sup_region == sub_region`.
    ///
    /// Panics if called before `solve()` executes,
    // This is `pub` because it's used by unstable external borrowck data users, see `consumers.rs`.
    pub fn eval_equal(&self, r1: RegionVid, r2: RegionVid) -> bool {
        self.eval_outlives(r1, r2) && self.eval_outlives(r2, r1)
    }

    /// Evaluate whether `sup_region: sub_region`.
    ///
    /// Panics if called before `solve()` executes,
    // This is `pub` because it's used by unstable external borrowck data users, see `consumers.rs`.
    #[instrument(skip(self), level = "debug", ret)]
    pub fn eval_outlives(&self, sup_region: RegionVid, sub_region: RegionVid) -> bool {
        debug!(
            "sup_region's value = {:?} universal={:?}",
            self.region_value_str(sup_region),
            self.universal_regions().is_universal_region(sup_region),
        );
        debug!(
            "sub_region's value = {:?} universal={:?}",
            self.region_value_str(sub_region),
            self.universal_regions().is_universal_region(sub_region),
        );

        let sub_region_scc = self.constraint_sccs.scc(sub_region);
        let sup_region_scc = self.constraint_sccs.scc(sup_region);

        if sub_region_scc == sup_region_scc {
            debug!("{sup_region:?}: {sub_region:?} holds trivially; they are in the same SCC");
            return true;
        }

        let fr_static = self.universal_regions().fr_static;

        // If we are checking that `'sup: 'sub`, and `'sub` contains
        // some placeholder that `'sup` cannot name, then this is only
        // true if `'sup` outlives static.
        //
        // Avoid infinite recursion if `sub_region` is already `'static`
        if sub_region != fr_static
            && !self.can_name_all_placeholders(sup_region_scc, sub_region_scc)
        {
            debug!(
                "sub universe `{sub_region_scc:?}` is not nameable \
                by super `{sup_region_scc:?}`, promoting to static",
            );

            return self.eval_outlives(sup_region, fr_static);
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

        if self.universal_regions().is_universal_region(sup_region) {
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
            debug!(?fr, ?fr_definition);
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
        polonius_output: &PoloniusOutput,
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

        for &(longer_fr, shorter_fr) in subset_errors.into_iter() {
            debug!(
                "check_polonius_subset_errors: subset_error longer_fr={:?},\
                 shorter_fr={:?}",
                longer_fr, shorter_fr
            );

            let propagated = self.try_propagate_universal_region_error(
                longer_fr.into(),
                shorter_fr.into(),
                &mut propagated_outlives_requirements,
            );
            if propagated == RegionRelationCheckResult::Error {
                errors_buffer.push(RegionErrorKind::RegionError {
                    longer_fr: longer_fr.into(),
                    shorter_fr: shorter_fr.into(),
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

    /// The largest universe of any region nameable from this SCC.
    fn max_nameable_universe(&self, scc: ConstraintSccIndex) -> UniverseIndex {
        self.scc_annotations[scc].max_nameable_universe()
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
        assert!(self.max_nameable_universe(longer_fr_scc).is_root());

        // Only check all of the relations for the main representative of each
        // SCC, otherwise just check that we outlive said representative. This
        // reduces the number of redundant relations propagated out of
        // closures.
        // Note that the representative will be a universal region if there is
        // one in this SCC, so we will always check the representative here.
        let representative = self.scc_representative(longer_fr_scc);
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
        if let Some(propagated_outlives_requirements) = propagated_outlives_requirements
            // Shrink `longer_fr` until we find a non-local region (if we do).
            // We'll call it `fr-` -- it's ever so slightly smaller than
            // `longer_fr`.
            && let Some(fr_minus) = self.universal_region_relations.non_local_lower_bound(longer_fr)
        {
            debug!("try_propagate_universal_region_error: fr_minus={:?}", fr_minus);

            let blame_constraint = self
                .best_blame_constraint(longer_fr, NllRegionVariableOrigin::FreeRegion, shorter_fr)
                .0;

            // Grow `shorter_fr` until we find some non-local regions. (We
            // always will.)  We'll call them `shorter_fr+` -- they're ever
            // so slightly larger than `shorter_fr`.
            let shorter_fr_plus =
                self.universal_region_relations.non_local_upper_bounds(shorter_fr);
            debug!("try_propagate_universal_region_error: shorter_fr_plus={:?}", shorter_fr_plus);
            for fr in shorter_fr_plus {
                // Push the constraint `fr-: shorter_fr+`
                propagated_outlives_requirements.push(ClosureOutlivesRequirement {
                    subject: ClosureOutlivesSubject::Region(fr_minus),
                    outlived_free_region: fr,
                    blame_span: blame_constraint.cause.span,
                    category: blame_constraint.category,
                });
            }
            return RegionRelationCheckResult::Propagated;
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

        // If we have some bound universal region `'a`, then the only
        // elements it can contain is itself -- we don't know anything
        // else about it!
        if let Some(error_element) = self
            .scc_values
            .elements_contained_in(longer_fr_scc)
            .find(|e| *e != RegionElement::PlaceholderRegion(placeholder))
        {
            // Stop after the first error, it gets too noisy otherwise, and does not provide more information.
            errors_buffer.push(RegionErrorKind::BoundUniversalRegionError {
                longer_fr,
                error_element,
                placeholder,
            });
        } else {
            debug!("check_bound_universal_region: all bounds satisfied");
        }
    }

    pub(crate) fn constraint_path_between_regions(
        &self,
        from_region: RegionVid,
        to_region: RegionVid,
    ) -> Option<Vec<OutlivesConstraint<'tcx>>> {
        if from_region == to_region {
            bug!("Tried to find a path between {from_region:?} and itself!");
        }
        self.constraint_path_to(from_region, |to| to == to_region, true).map(|o| o.0)
    }

    /// Walks the graph of constraints (where `'a: 'b` is considered
    /// an edge `'a -> 'b`) to find a path from `from_region` to
    /// `to_region`.
    ///
    /// Returns: a series of constraints as well as the region `R`
    /// that passed the target test.
    /// If `include_static_outlives_all` is `true`, then the synthetic
    /// outlives constraints `'static -> a` for every region `a` are
    /// considered in the search, otherwise they are ignored.
    #[instrument(skip(self, target_test), ret)]
    pub(crate) fn constraint_path_to(
        &self,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
        include_placeholder_static: bool,
    ) -> Option<(Vec<OutlivesConstraint<'tcx>>, RegionVid)> {
        self.find_constraint_path_between_regions_inner(
            true,
            from_region,
            &target_test,
            include_placeholder_static,
        )
        .or_else(|| {
            self.find_constraint_path_between_regions_inner(
                false,
                from_region,
                &target_test,
                include_placeholder_static,
            )
        })
    }

    /// The constraints we get from equating the hidden type of each use of an opaque
    /// with its final hidden type may end up getting preferred over other, potentially
    /// longer constraint paths.
    ///
    /// Given that we compute the final hidden type by relying on this existing constraint
    /// path, this can easily end up hiding the actual reason for why we require these regions
    /// to be equal.
    ///
    /// To handle this, we first look at the path while ignoring these constraints and then
    /// retry while considering them. This is not perfect, as the `from_region` may have already
    /// been partially related to its argument region, so while we rely on a member constraint
    /// to get a complete path, the most relevant step of that path already existed before then.
    fn find_constraint_path_between_regions_inner(
        &self,
        ignore_opaque_type_constraints: bool,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
        include_placeholder_static: bool,
    ) -> Option<(Vec<OutlivesConstraint<'tcx>>, RegionVid)> {
        let mut context = IndexVec::from_elem(Trace::NotVisited, &self.definitions);
        context[from_region] = Trace::StartRegion;

        let fr_static = self.universal_regions().fr_static;

        // Use a deque so that we do a breadth-first search. We will
        // stop at the first match, which ought to be the shortest
        // path (fewest constraints).
        let mut deque = VecDeque::new();
        deque.push_back(from_region);

        while let Some(r) = deque.pop_front() {
            debug!(
                "constraint_path_to: from_region={:?} r={:?} value={}",
                from_region,
                r,
                self.region_value_str(r),
            );

            // Check if we reached the region we were looking for. If so,
            // we can reconstruct the path that led to it and return it.
            if target_test(r) {
                let mut result = vec![];
                let mut p = r;
                // This loop is cold and runs at the end, which is why we delay
                // `OutlivesConstraint` construction until now.
                loop {
                    match context[p] {
                        Trace::FromGraph(c) => {
                            p = c.sup;
                            result.push(*c);
                        }

                        Trace::FromStatic(sub) => {
                            let c = OutlivesConstraint {
                                sup: fr_static,
                                sub,
                                locations: Locations::All(DUMMY_SP),
                                span: DUMMY_SP,
                                category: ConstraintCategory::Internal,
                                variance_info: ty::VarianceDiagInfo::default(),
                                from_closure: false,
                            };
                            p = c.sup;
                            result.push(c);
                        }

                        Trace::StartRegion => {
                            result.reverse();
                            return Some((result, r));
                        }

                        Trace::NotVisited => {
                            bug!("found unvisited region {:?} on path to {:?}", p, r)
                        }
                    }
                }
            }

            // Otherwise, walk over the outgoing constraints and
            // enqueue any regions we find, keeping track of how we
            // reached them.

            // A constraint like `'r: 'x` can come from our constraint
            // graph.

            // Always inline this closure because it can be hot.
            let mut handle_trace = #[inline(always)]
            |sub, trace| {
                if let Trace::NotVisited = context[sub] {
                    context[sub] = trace;
                    deque.push_back(sub);
                }
            };

            // If this is the `'static` region and the graph's direction is normal, then set up the
            // Edges iterator to return all regions (#53178).
            if r == fr_static && self.constraint_graph.is_normal() {
                for sub in self.constraint_graph.outgoing_edges_from_static() {
                    handle_trace(sub, Trace::FromStatic(sub));
                }
            } else {
                let edges = self.constraint_graph.outgoing_edges_from_graph(r, &self.constraints);
                // This loop can be hot.
                for constraint in edges {
                    match constraint.category {
                        ConstraintCategory::OutlivesUnnameablePlaceholder(_)
                            if !include_placeholder_static =>
                        {
                            debug!("Ignoring illegal placeholder constraint: {constraint:?}");
                            continue;
                        }
                        ConstraintCategory::OpaqueType if ignore_opaque_type_constraints => {
                            debug!("Ignoring member constraint: {constraint:?}");
                            continue;
                        }
                        _ => {}
                    }

                    debug_assert_eq!(constraint.sup, r);
                    handle_trace(constraint.sub, Trace::FromGraph(constraint));
                }
            }
        }

        None
    }

    /// Finds some region R such that `fr1: R` and `R` is live at `location`.
    #[instrument(skip(self), level = "trace", ret)]
    pub(crate) fn find_sub_region_live_at(&self, fr1: RegionVid, location: Location) -> RegionVid {
        trace!(scc = ?self.constraint_sccs.scc(fr1));
        trace!(universe = ?self.max_nameable_universe(self.constraint_sccs.scc(fr1)));
        self.constraint_path_to(fr1, |r| {
            trace!(?r, liveness_constraints=?self.liveness_constraints.pretty_print_live_points(r));
            self.liveness_constraints.is_live_at(r, location)
        }, true).unwrap().1
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
        &self.universal_region_relations.universal_regions
    }

    /// Tries to find the best constraint to blame for the fact that
    /// `R: from_region`, where `R` is some region that meets
    /// `target_test`. This works by following the constraint graph,
    /// creating a constraint path that forces `R` to outlive
    /// `from_region`, and then finding the best choices within that
    /// path to blame.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn best_blame_constraint(
        &self,
        from_region: RegionVid,
        from_region_origin: NllRegionVariableOrigin,
        to_region: RegionVid,
    ) -> (BlameConstraint<'tcx>, Vec<OutlivesConstraint<'tcx>>) {
        assert!(from_region != to_region, "Trying to blame a region for itself!");

        let path = self.constraint_path_between_regions(from_region, to_region).unwrap();

        // If we are passing through a constraint added because we reached an unnameable placeholder `'unnameable`,
        // redirect search towards `'unnameable`.
        let due_to_placeholder_outlives = path.iter().find_map(|c| {
            if let ConstraintCategory::OutlivesUnnameablePlaceholder(unnameable) = c.category {
                Some(unnameable)
            } else {
                None
            }
        });

        // Edge case: it's possible that `'from_region` is an unnameable placeholder.
        let path = if let Some(unnameable) = due_to_placeholder_outlives
            && unnameable != from_region
        {
            // We ignore the extra edges due to unnameable placeholders to get
            // an explanation that was present in the original constraint graph.
            self.constraint_path_to(from_region, |r| r == unnameable, false).unwrap().0
        } else {
            path
        };

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

        // We try to avoid reporting a `ConstraintCategory::Predicate` as our best constraint.
        // Instead, we use it to produce an improved `ObligationCauseCode`.
        // FIXME - determine what we should do if we encounter multiple
        // `ConstraintCategory::Predicate` constraints. Currently, we just pick the first one.
        let cause_code = path
            .iter()
            .find_map(|constraint| {
                if let ConstraintCategory::Predicate(predicate_span) = constraint.category {
                    // We currently do not store the `DefId` in the `ConstraintCategory`
                    // for performances reasons. The error reporting code used by NLL only
                    // uses the span, so this doesn't cause any problems at the moment.
                    Some(ObligationCauseCode::WhereClause(CRATE_DEF_ID.to_def_id(), predicate_span))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| ObligationCauseCode::Misc);

        // When reporting an error, there is typically a chain of constraints leading from some
        // "source" region which must outlive some "target" region.
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
            NllRegionVariableOrigin::FreeRegion => true,
            NllRegionVariableOrigin::Placeholder(_) => false,
            // `'existential: 'whatever` never results in a region error by itself.
            // We may always infer it to `'static` afterall. This means while an error
            // path may go through an existential, these existentials are never the
            // `from_region`.
            NllRegionVariableOrigin::Existential { name: _ } => {
                unreachable!("existentials can outlive everything")
            }
        };

        // To pick a constraint to blame, we organize constraints by how interesting we expect them
        // to be in diagnostics, then pick the most interesting one closest to either the source or
        // the target on our constraint path.
        let constraint_interest = |constraint: &OutlivesConstraint<'tcx>| {
            // Try to avoid blaming constraints from desugarings, since they may not clearly match
            // match what users have written. As an exception, allow blaming returns generated by
            // `?` desugaring, since the correspondence is fairly clear.
            let category = if let Some(kind) = constraint.span.desugaring_kind()
                && (kind != DesugaringKind::QuestionMark
                    || !matches!(constraint.category, ConstraintCategory::Return(_)))
            {
                ConstraintCategory::Boring
            } else {
                constraint.category
            };

            let interest = match category {
                // Returns usually provide a type to blame and have specially written diagnostics,
                // so prioritize them.
                ConstraintCategory::Return(_) => 0,
                // Unsizing coercions are interesting, since we have a note for that:
                // `BorrowExplanation::add_object_lifetime_default_note`.
                // FIXME(dianne): That note shouldn't depend on a coercion being blamed; see issue
                // #131008 for an example of where we currently don't emit it but should.
                // Once the note is handled properly, this case should be removed. Until then, it
                // should be as limited as possible; the note is prone to false positives and this
                // constraint usually isn't best to blame.
                ConstraintCategory::Cast {
                    unsize_to: Some(unsize_ty),
                    is_implicit_coercion: true,
                } if to_region == self.universal_regions().fr_static
                    // Mirror the note's condition, to minimize how often this diverts blame.
                    && let ty::Adt(_, args) = unsize_ty.kind()
                    && args.iter().any(|arg| arg.as_type().is_some_and(|ty| ty.is_trait()))
                    // Mimic old logic for this, to minimize false positives in tests.
                    && !path
                        .iter()
                        .any(|c| matches!(c.category, ConstraintCategory::TypeAnnotation(_))) =>
                {
                    1
                }
                // Between other interesting constraints, order by their position on the `path`.
                ConstraintCategory::Yield
                | ConstraintCategory::UseAsConst
                | ConstraintCategory::UseAsStatic
                | ConstraintCategory::TypeAnnotation(
                    AnnotationSource::Ascription
                    | AnnotationSource::Declaration
                    | AnnotationSource::OpaqueCast,
                )
                | ConstraintCategory::Cast { .. }
                | ConstraintCategory::CallArgument(_)
                | ConstraintCategory::CopyBound
                | ConstraintCategory::SizedBound
                | ConstraintCategory::Assignment
                | ConstraintCategory::Usage
                | ConstraintCategory::ClosureUpvar(_) => 2,
                // Generic arguments are unlikely to be what relates regions together
                ConstraintCategory::TypeAnnotation(AnnotationSource::GenericArg) => 3,
                // We handle predicates and opaque types specially; don't prioritize them here.
                ConstraintCategory::Predicate(_) | ConstraintCategory::OpaqueType => 4,
                // `Boring` constraints can correspond to user-written code and have useful spans,
                // but don't provide any other useful information for diagnostics.
                ConstraintCategory::Boring => 5,
                // `BoringNoLocation` constraints can point to user-written code, but are less
                // specific, and are not used for relations that would make sense to blame.
                ConstraintCategory::BoringNoLocation => 6,
                // Do not blame internal constraints if we can avoid it. Never blame
                // the `'region: 'static` constraints introduced by placeholder outlives.
                ConstraintCategory::Internal => 7,
                ConstraintCategory::OutlivesUnnameablePlaceholder(_) => 8,
            };

            debug!("constraint {constraint:?} category: {category:?}, interest: {interest:?}");

            interest
        };

        let best_choice = if blame_source {
            path.iter().enumerate().rev().min_by_key(|(_, c)| constraint_interest(c)).unwrap().0
        } else {
            path.iter().enumerate().min_by_key(|(_, c)| constraint_interest(c)).unwrap().0
        };

        debug!(?best_choice, ?blame_source);

        let best_constraint = if let Some(next) = path.get(best_choice + 1)
            && matches!(path[best_choice].category, ConstraintCategory::Return(_))
            && next.category == ConstraintCategory::OpaqueType
        {
            // The return expression is being influenced by the return type being
            // impl Trait, point at the return type and not the return expr.
            *next
        } else if path[best_choice].category == ConstraintCategory::Return(ReturnConstraint::Normal)
            && let Some(field) = path.iter().find_map(|p| {
                if let ConstraintCategory::ClosureUpvar(f) = p.category { Some(f) } else { None }
            })
        {
            OutlivesConstraint {
                category: ConstraintCategory::Return(ReturnConstraint::ClosureUpvar(field)),
                ..path[best_choice]
            }
        } else {
            path[best_choice]
        };

        assert!(
            !matches!(
                best_constraint.category,
                ConstraintCategory::OutlivesUnnameablePlaceholder(_)
            ),
            "Illegal placeholder constraint blamed; should have redirected to other region relation"
        );

        let blame_constraint = BlameConstraint {
            category: best_constraint.category,
            from_closure: best_constraint.from_closure,
            cause: ObligationCause::new(best_constraint.span, CRATE_DEF_ID, cause_code.clone()),
            variance_info: best_constraint.variance_info,
        };
        (blame_constraint, path)
    }

    pub(crate) fn universe_info(&self, universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        // Query canonicalization can create local superuniverses (for example in
        // `InferCtx::query_response_instantiation_guess`), but they don't have an associated
        // `UniverseInfo` explaining why they were created.
        // This can cause ICEs if these causes are accessed in diagnostics, for example in issue
        // #114907 where this happens via liveness and dropck outlives results.
        // Therefore, we return a default value in case that happens, which should at worst emit a
        // suboptimal error, instead of the ICE.
        self.universe_causes.get(&universe).cloned().unwrap_or_else(UniverseInfo::other)
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
            if let Some(terminator) = &bb.terminator
                // terminator of a loop should be TerminatorKind::FalseUnwind
                && let TerminatorKind::FalseUnwind { .. } = terminator.kind
            {
                return Some(location);
            }
        }
        None
    }

    /// Access to the SCC constraint graph.
    /// This can be used to quickly under-approximate the regions which are equal to each other
    /// and their relative orderings.
    // This is `pub` because it's used by unstable external borrowck data users, see `consumers.rs`.
    pub fn constraint_sccs(&self) -> &ConstraintSccs {
        &self.constraint_sccs
    }

    /// Returns the representative `RegionVid` for a given SCC.
    /// See `RegionTracker` for how a region variable ID is chosen.
    ///
    /// It is a hacky way to manage checking regions for equality,
    /// since we can 'canonicalize' each region to the representative
    /// of its SCC and be sure that -- if they have the same repr --
    /// they *must* be equal (though not having the same repr does not
    /// mean they are unequal).
    fn scc_representative(&self, scc: ConstraintSccIndex) -> RegionVid {
        self.scc_annotations[scc].representative.rvid()
    }

    pub(crate) fn liveness_constraints(&self) -> &LivenessValues {
        &self.liveness_constraints
    }

    /// When using `-Zpolonius=next`, records the given live loans for the loan scopes and active
    /// loans dataflow computations.
    pub(crate) fn record_live_loans(&mut self, live_loans: LiveLoans) {
        self.liveness_constraints.record_live_loans(live_loans);
    }

    /// Returns whether the `loan_idx` is live at the given `location`: whether its issuing
    /// region is contained within the type of a variable that is live at this point.
    /// Note: for now, the sets of live loans is only available when using `-Zpolonius=next`.
    pub(crate) fn is_loan_live_at(&self, loan_idx: BorrowIndex, location: Location) -> bool {
        let point = self.liveness_constraints.point_from_location(location);
        self.liveness_constraints.is_loan_live_at(loan_idx, point)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BlameConstraint<'tcx> {
    pub category: ConstraintCategory<'tcx>,
    pub from_closure: bool,
    pub cause: ObligationCause<'tcx>,
    pub variance_info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
}
