//! This pass type-checks the MIR to ensure it is not broken.

use std::rc::Rc;
use std::{fmt, iter, mem};

use rustc_abi::FieldIdx;
use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::lang_items::LangItem;
use rustc_index::{IndexSlice, IndexVec};
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::infer::outlives::env::RegionBoundPairs;
use rustc_infer::infer::region_constraints::RegionConstraintData;
use rustc_infer::infer::{
    BoundRegion, BoundRegionConversionTime, InferCtxt, NllRegionVariableOrigin,
};
use rustc_infer::traits::PredicateObligations;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::cast::CastTy;
use rustc_middle::ty::{
    self, Binder, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations, CoroutineArgsExt,
    Dynamic, GenericArgsRef, OpaqueHiddenType, OpaqueTypeKey, RegionVid, Ty, TyCtxt,
    TypeVisitableExt, UserArgs, UserTypeAnnotationIndex, fold_regions,
};
use rustc_middle::{bug, span_bug};
use rustc_mir_dataflow::ResultsCursor;
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::points::DenseLocationMap;
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, sym};
use rustc_trait_selection::traits::query::type_op::custom::scrape_region_constraints;
use rustc_trait_selection::traits::query::type_op::{TypeOp, TypeOpOutput};
use tracing::{debug, instrument, trace};

use crate::borrow_set::BorrowSet;
use crate::constraints::{OutlivesConstraint, OutlivesConstraintSet};
use crate::diagnostics::UniverseInfo;
use crate::member_constraints::MemberConstraintSet;
use crate::polonius::legacy::{PoloniusFacts, PoloniusLocationTable};
use crate::polonius::{PoloniusContext, PoloniusLivenessContext};
use crate::region_infer::TypeTest;
use crate::region_infer::values::{LivenessValues, PlaceholderIndex, PlaceholderIndices};
use crate::session_diagnostics::{MoveUnsized, SimdIntrinsicArgConst};
use crate::type_check::free_region_relations::{CreateResult, UniversalRegionRelations};
use crate::universal_regions::{DefiningTy, UniversalRegions};
use crate::{BorrowCheckRootCtxt, BorrowckInferCtxt, path_utils};

macro_rules! span_mirbug {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        $crate::type_check::mirbug(
            $context.tcx(),
            $context.last_span,
            format!(
                "broken MIR in {:?} ({:?}): {}",
                $context.body().source.def_id(),
                $elem,
                format_args!($($message)*),
            ),
        )
    })
}

mod canonical;
mod constraint_conversion;
pub(crate) mod free_region_relations;
mod input_output;
pub(crate) mod liveness;
mod opaque_types;
mod relate_tys;

/// Type checks the given `mir` in the context of the inference
/// context `infcx`. Returns any region constraints that have yet to
/// be proven. This result includes liveness constraints that
/// ensure that regions appearing in the types of all local variables
/// are live at all points where that local variable may later be
/// used.
///
/// This phase of type-check ought to be infallible -- this is because
/// the original, HIR-based type-check succeeded. So if any errors
/// occur here, we will get a `bug!` reported.
///
/// # Parameters
///
/// - `infcx` -- inference context to use
/// - `body` -- MIR body to type-check
/// - `promoted` -- map of promoted constants within `body`
/// - `universal_regions` -- the universal regions from `body`s function signature
/// - `location_table` -- for datalog polonius, the map between `Location`s and `RichLocation`s
/// - `borrow_set` -- information about borrows occurring in `body`
/// - `polonius_facts` -- when using Polonius, this is the generated set of Polonius facts
/// - `flow_inits` -- results of a maybe-init dataflow analysis
/// - `move_data` -- move-data constructed when performing the maybe-init dataflow analysis
/// - `location_map` -- map between MIR `Location` and `PointIndex`
pub(crate) fn type_check<'a, 'tcx>(
    root_cx: &mut BorrowCheckRootCtxt<'tcx>,
    infcx: &BorrowckInferCtxt<'tcx>,
    body: &Body<'tcx>,
    promoted: &IndexSlice<Promoted, Body<'tcx>>,
    universal_regions: UniversalRegions<'tcx>,
    location_table: &PoloniusLocationTable,
    borrow_set: &BorrowSet<'tcx>,
    polonius_facts: &mut Option<PoloniusFacts>,
    flow_inits: ResultsCursor<'a, 'tcx, MaybeInitializedPlaces<'a, 'tcx>>,
    move_data: &MoveData<'tcx>,
    location_map: Rc<DenseLocationMap>,
) -> MirTypeckResults<'tcx> {
    let implicit_region_bound = ty::Region::new_var(infcx.tcx, universal_regions.fr_fn_body);
    let mut constraints = MirTypeckRegionConstraints {
        placeholder_indices: PlaceholderIndices::default(),
        placeholder_index_to_region: IndexVec::default(),
        liveness_constraints: LivenessValues::with_specific_points(Rc::clone(&location_map)),
        outlives_constraints: OutlivesConstraintSet::default(),
        member_constraints: MemberConstraintSet::default(),
        type_tests: Vec::default(),
        universe_causes: FxIndexMap::default(),
    };

    let CreateResult {
        universal_region_relations,
        region_bound_pairs,
        normalized_inputs_and_output,
        known_type_outlives_obligations,
    } = free_region_relations::create(
        infcx,
        infcx.param_env,
        implicit_region_bound,
        universal_regions,
        &mut constraints,
    );

    let pre_obligations = infcx.take_registered_region_obligations();
    assert!(
        pre_obligations.is_empty(),
        "there should be no incoming region obligations = {pre_obligations:#?}",
    );

    debug!(?normalized_inputs_and_output);

    let polonius_liveness = if infcx.tcx.sess.opts.unstable_opts.polonius.is_next_enabled() {
        Some(PoloniusLivenessContext::default())
    } else {
        None
    };

    let mut typeck = TypeChecker {
        root_cx,
        infcx,
        last_span: body.span,
        body,
        promoted,
        user_type_annotations: &body.user_type_annotations,
        region_bound_pairs,
        known_type_outlives_obligations,
        implicit_region_bound,
        reported_errors: Default::default(),
        universal_regions: &universal_region_relations.universal_regions,
        location_table,
        polonius_facts,
        borrow_set,
        constraints: &mut constraints,
        polonius_liveness,
    };

    typeck.check_user_type_annotations();
    typeck.visit_body(body);
    typeck.equate_inputs_and_outputs(&normalized_inputs_and_output);
    typeck.check_signature_annotation();

    liveness::generate(&mut typeck, &location_map, flow_inits, move_data);

    let opaque_type_values =
        opaque_types::take_opaques_and_register_member_constraints(&mut typeck);

    // We're done with typeck, we can finalize the polonius liveness context for region inference.
    let polonius_context = typeck.polonius_liveness.take().map(|liveness_context| {
        PoloniusContext::create_from_liveness(
            liveness_context,
            infcx.num_region_vars(),
            typeck.constraints.liveness_constraints.points(),
        )
    });

    MirTypeckResults {
        constraints,
        universal_region_relations,
        opaque_type_values,
        polonius_context,
    }
}

#[track_caller]
fn mirbug(tcx: TyCtxt<'_>, span: Span, msg: String) {
    // We sometimes see MIR failures (notably predicate failures) due to
    // the fact that we check rvalue sized predicates here. So use `span_delayed_bug`
    // to avoid reporting bugs in those cases.
    tcx.dcx().span_delayed_bug(span, msg);
}

enum FieldAccessError {
    OutOfRange { field_count: usize },
}

/// The MIR type checker. Visits the MIR and enforces all the
/// constraints needed for it to be valid and well-typed. Along the
/// way, it accrues region constraints -- these can later be used by
/// NLL region checking.
struct TypeChecker<'a, 'tcx> {
    root_cx: &'a mut BorrowCheckRootCtxt<'tcx>,
    infcx: &'a BorrowckInferCtxt<'tcx>,
    last_span: Span,
    body: &'a Body<'tcx>,
    /// The bodies of all promoteds. As promoteds have a completely separate CFG
    /// recursing into them may corrupt your data structures if you're not careful.
    promoted: &'a IndexSlice<Promoted, Body<'tcx>>,
    /// User type annotations are shared between the main MIR and the MIR of
    /// all of the promoted items.
    user_type_annotations: &'a CanonicalUserTypeAnnotations<'tcx>,
    region_bound_pairs: RegionBoundPairs<'tcx>,
    known_type_outlives_obligations: Vec<ty::PolyTypeOutlivesPredicate<'tcx>>,
    implicit_region_bound: ty::Region<'tcx>,
    reported_errors: FxIndexSet<(Ty<'tcx>, Span)>,
    universal_regions: &'a UniversalRegions<'tcx>,
    location_table: &'a PoloniusLocationTable,
    polonius_facts: &'a mut Option<PoloniusFacts>,
    borrow_set: &'a BorrowSet<'tcx>,
    constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
    /// When using `-Zpolonius=next`, the liveness helper data used to create polonius constraints.
    polonius_liveness: Option<PoloniusLivenessContext>,
}

/// Holder struct for passing results from MIR typeck to the rest of the non-lexical regions
/// inference computation.
pub(crate) struct MirTypeckResults<'tcx> {
    pub(crate) constraints: MirTypeckRegionConstraints<'tcx>,
    pub(crate) universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
    pub(crate) opaque_type_values: FxIndexMap<OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>>,
    pub(crate) polonius_context: Option<PoloniusContext>,
}

/// A collection of region constraints that must be satisfied for the
/// program to be considered well-typed.
pub(crate) struct MirTypeckRegionConstraints<'tcx> {
    /// Maps from a `ty::Placeholder` to the corresponding
    /// `PlaceholderIndex` bit that we will use for it.
    ///
    /// To keep everything in sync, do not insert this set
    /// directly. Instead, use the `placeholder_region` helper.
    pub(crate) placeholder_indices: PlaceholderIndices,

    /// Each time we add a placeholder to `placeholder_indices`, we
    /// also create a corresponding "representative" region vid for
    /// that wraps it. This vector tracks those. This way, when we
    /// convert the same `ty::RePlaceholder(p)` twice, we can map to
    /// the same underlying `RegionVid`.
    pub(crate) placeholder_index_to_region: IndexVec<PlaceholderIndex, ty::Region<'tcx>>,

    /// In general, the type-checker is not responsible for enforcing
    /// liveness constraints; this job falls to the region inferencer,
    /// which performs a liveness analysis. However, in some limited
    /// cases, the MIR type-checker creates temporary regions that do
    /// not otherwise appear in the MIR -- in particular, the
    /// late-bound regions that it instantiates at call-sites -- and
    /// hence it must report on their liveness constraints.
    pub(crate) liveness_constraints: LivenessValues,

    pub(crate) outlives_constraints: OutlivesConstraintSet<'tcx>,

    pub(crate) member_constraints: MemberConstraintSet<'tcx, RegionVid>,

    pub(crate) universe_causes: FxIndexMap<ty::UniverseIndex, UniverseInfo<'tcx>>,

    pub(crate) type_tests: Vec<TypeTest<'tcx>>,
}

impl<'tcx> MirTypeckRegionConstraints<'tcx> {
    /// Creates a `Region` for a given `PlaceholderRegion`, or returns the
    /// region that corresponds to a previously created one.
    fn placeholder_region(
        &mut self,
        infcx: &InferCtxt<'tcx>,
        placeholder: ty::PlaceholderRegion,
    ) -> ty::Region<'tcx> {
        let placeholder_index = self.placeholder_indices.insert(placeholder);
        match self.placeholder_index_to_region.get(placeholder_index) {
            Some(&v) => v,
            None => {
                let origin = NllRegionVariableOrigin::Placeholder(placeholder);
                let region = infcx.next_nll_region_var_in_universe(origin, placeholder.universe);
                self.placeholder_index_to_region.push(region);
                region
            }
        }
    }
}

/// The `Locations` type summarizes *where* region constraints are
/// required to hold. Normally, this is at a particular point which
/// created the obligation, but for constraints that the user gave, we
/// want the constraint to hold at all points.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Locations {
    /// Indicates that a type constraint should always be true. This
    /// is particularly important in the new borrowck analysis for
    /// things like the type of the return slot. Consider this
    /// example:
    ///
    /// ```compile_fail,E0515
    /// fn foo<'a>(x: &'a u32) -> &'a u32 {
    ///     let y = 22;
    ///     return &y; // error
    /// }
    /// ```
    ///
    /// Here, we wind up with the signature from the return type being
    /// something like `&'1 u32` where `'1` is a universal region. But
    /// the type of the return slot `_0` is something like `&'2 u32`
    /// where `'2` is an existential region variable. The type checker
    /// requires that `&'2 u32 = &'1 u32` -- but at what point? In the
    /// older NLL analysis, we required this only at the entry point
    /// to the function. By the nature of the constraints, this wound
    /// up propagating to all points reachable from start (because
    /// `'1` -- as a universal region -- is live everywhere). In the
    /// newer analysis, though, this doesn't work: `_0` is considered
    /// dead at the start (it has no usable value) and hence this type
    /// equality is basically a no-op. Then, later on, when we do `_0
    /// = &'3 y`, that region `'3` never winds up related to the
    /// universal region `'1` and hence no error occurs. Therefore, we
    /// use Locations::All instead, which ensures that the `'1` and
    /// `'2` are equal everything. We also use this for other
    /// user-given type annotations; e.g., if the user wrote `let mut
    /// x: &'static u32 = ...`, we would ensure that all values
    /// assigned to `x` are of `'static` lifetime.
    ///
    /// The span points to the place the constraint arose. For example,
    /// it points to the type in a user-given type annotation. If
    /// there's no sensible span then it's DUMMY_SP.
    All(Span),

    /// An outlives constraint that only has to hold at a single location,
    /// usually it represents a point where references flow from one spot to
    /// another (e.g., `x = y`)
    Single(Location),
}

impl Locations {
    pub fn from_location(&self) -> Option<Location> {
        match self {
            Locations::All(_) => None,
            Locations::Single(from_location) => Some(*from_location),
        }
    }

    /// Gets a span representing the location.
    pub fn span(&self, body: &Body<'_>) -> Span {
        match self {
            Locations::All(span) => *span,
            Locations::Single(l) => body.source_info(*l).span,
        }
    }
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn body(&self) -> &Body<'tcx> {
        self.body
    }

    fn to_region_vid(&mut self, r: ty::Region<'tcx>) -> RegionVid {
        if let ty::RePlaceholder(placeholder) = r.kind() {
            self.constraints.placeholder_region(self.infcx, placeholder).as_var()
        } else {
            self.universal_regions.to_region_vid(r)
        }
    }

    fn unsized_feature_enabled(&self) -> bool {
        let features = self.tcx().features();
        features.unsized_locals() || features.unsized_fn_params()
    }

    /// Equate the inferred type and the annotated type for user type annotations
    #[instrument(skip(self), level = "debug")]
    fn check_user_type_annotations(&mut self) {
        debug!(?self.user_type_annotations);
        let tcx = self.tcx();
        for user_annotation in self.user_type_annotations {
            let CanonicalUserTypeAnnotation { span, ref user_ty, inferred_ty } = *user_annotation;
            let annotation = self.instantiate_canonical(span, user_ty);
            if let ty::UserTypeKind::TypeOf(def, args) = annotation.kind
                && let DefKind::InlineConst = tcx.def_kind(def)
            {
                assert!(annotation.bounds.is_empty());
                self.check_inline_const(inferred_ty, def.expect_local(), args, span);
            } else {
                self.ascribe_user_type(inferred_ty, annotation, span);
            }
        }
    }

    #[instrument(skip(self, data), level = "debug")]
    fn push_region_constraints(
        &mut self,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
        data: &QueryRegionConstraints<'tcx>,
    ) {
        debug!("constraints generated: {:#?}", data);

        constraint_conversion::ConstraintConversion::new(
            self.infcx,
            self.universal_regions,
            &self.region_bound_pairs,
            self.implicit_region_bound,
            self.infcx.param_env,
            &self.known_type_outlives_obligations,
            locations,
            locations.span(self.body),
            category,
            self.constraints,
        )
        .convert_all(data);
    }

    /// Try to relate `sub <: sup`
    fn sub_types(
        &mut self,
        sub: Ty<'tcx>,
        sup: Ty<'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) -> Result<(), NoSolution> {
        // Use this order of parameters because the sup type is usually the
        // "expected" type in diagnostics.
        self.relate_types(sup, ty::Contravariant, sub, locations, category)
    }

    #[instrument(skip(self, category), level = "debug")]
    fn eq_types(
        &mut self,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) -> Result<(), NoSolution> {
        self.relate_types(expected, ty::Invariant, found, locations, category)
    }

    #[instrument(skip(self), level = "debug")]
    fn relate_type_and_user_type(
        &mut self,
        a: Ty<'tcx>,
        v: ty::Variance,
        user_ty: &UserTypeProjection,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) -> Result<(), NoSolution> {
        let annotated_type = self.user_type_annotations[user_ty.base].inferred_ty;
        trace!(?annotated_type);
        let mut curr_projected_ty = PlaceTy::from_ty(annotated_type);

        let tcx = self.infcx.tcx;

        for proj in &user_ty.projs {
            if !self.infcx.next_trait_solver()
                && let ty::Alias(ty::Opaque, ..) = curr_projected_ty.ty.kind()
            {
                // There is nothing that we can compare here if we go through an opaque type.
                // We're always in its defining scope as we can otherwise not project through
                // it, so we're constraining it anyways.
                return Ok(());
            }
            let projected_ty = curr_projected_ty.projection_ty_core(
                tcx,
                proj,
                |this, field, ()| {
                    let ty = this.field_ty(tcx, field);
                    self.structurally_resolve(ty, locations)
                },
                |_, _| unreachable!(),
            );
            curr_projected_ty = projected_ty;
        }
        trace!(?curr_projected_ty);

        let ty = curr_projected_ty.ty;
        self.relate_types(ty, v.xform(ty::Contravariant), a, locations, category)?;

        Ok(())
    }

    fn check_promoted(&mut self, promoted_body: &'a Body<'tcx>, location: Location) {
        // Determine the constraints from the promoted MIR by running the type
        // checker on the promoted MIR, then transfer the constraints back to
        // the main MIR, changing the locations to the provided location.

        let parent_body = mem::replace(&mut self.body, promoted_body);

        // Use new sets of constraints and closure bounds so that we can
        // modify their locations.
        let polonius_facts = &mut None;
        let mut constraints = Default::default();
        let mut liveness_constraints =
            LivenessValues::without_specific_points(Rc::new(DenseLocationMap::new(promoted_body)));

        // Don't try to add borrow_region facts for the promoted MIR as they refer
        // to the wrong locations.
        let mut swap_constraints = |this: &mut Self| {
            mem::swap(this.polonius_facts, polonius_facts);
            mem::swap(&mut this.constraints.outlives_constraints, &mut constraints);
            mem::swap(&mut this.constraints.liveness_constraints, &mut liveness_constraints);
        };

        swap_constraints(self);

        self.visit_body(promoted_body);

        self.body = parent_body;

        // Merge the outlives constraints back in, at the given location.
        swap_constraints(self);
        let locations = location.to_locations();
        for constraint in constraints.outlives().iter() {
            let mut constraint = *constraint;
            constraint.locations = locations;
            if let ConstraintCategory::Return(_)
            | ConstraintCategory::UseAsConst
            | ConstraintCategory::UseAsStatic = constraint.category
            {
                // "Returning" from a promoted is an assignment to a
                // temporary from the user's point of view.
                constraint.category = ConstraintCategory::Boring;
            }
            self.constraints.outlives_constraints.push(constraint)
        }
        // If the region is live at least one location in the promoted MIR,
        // then add a liveness constraint to the main MIR for this region
        // at the location provided as an argument to this method
        //
        // add_location doesn't care about ordering so not a problem for the live regions to be
        // unordered.
        #[allow(rustc::potential_query_instability)]
        for region in liveness_constraints.live_regions_unordered() {
            self.constraints.liveness_constraints.add_location(region, location);
        }
    }

    fn check_inline_const(
        &mut self,
        inferred_ty: Ty<'tcx>,
        def_id: LocalDefId,
        args: UserArgs<'tcx>,
        span: Span,
    ) {
        assert!(args.user_self_ty.is_none());
        let tcx = self.tcx();
        let const_ty = tcx.type_of(def_id).instantiate(tcx, args.args);
        if let Err(terr) =
            self.eq_types(const_ty, inferred_ty, Locations::All(span), ConstraintCategory::Boring)
        {
            span_bug!(
                span,
                "bad inline const pattern: ({:?} = {:?}) {:?}",
                const_ty,
                inferred_ty,
                terr
            );
        }
        let args = self.infcx.resolve_vars_if_possible(args.args);
        let predicates = self.prove_closure_bounds(tcx, def_id, args, Locations::All(span));
        self.normalize_and_prove_instantiated_predicates(
            def_id.to_def_id(),
            predicates,
            Locations::All(span),
        );
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypeChecker<'a, 'tcx> {
    fn visit_span(&mut self, span: Span) {
        if !span.is_dummy() {
            debug!(?span);
            self.last_span = span;
        }
    }

    #[instrument(skip(self, body), level = "debug")]
    fn visit_body(&mut self, body: &Body<'tcx>) {
        debug_assert!(std::ptr::eq(self.body, body));

        for (local, local_decl) in body.local_decls.iter_enumerated() {
            self.visit_local_decl(local, local_decl);
        }

        for (block, block_data) in body.basic_blocks.iter_enumerated() {
            let mut location = Location { block, statement_index: 0 };
            for stmt in &block_data.statements {
                self.visit_statement(stmt, location);
                location.statement_index += 1;
            }

            self.visit_terminator(block_data.terminator(), location);
            self.check_iscleanup(block_data);
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_statement(&mut self, stmt: &Statement<'tcx>, location: Location) {
        self.super_statement(stmt, location);
        let tcx = self.tcx();
        match &stmt.kind {
            StatementKind::Assign(box (place, rv)) => {
                // Assignments to temporaries are not "interesting";
                // they are not caused by the user, but rather artifacts
                // of lowering. Assignments to other sorts of places *are* interesting
                // though.
                let category = match place.as_local() {
                    Some(RETURN_PLACE) => {
                        let defining_ty = &self.universal_regions.defining_ty;
                        if defining_ty.is_const() {
                            if tcx.is_static(defining_ty.def_id()) {
                                ConstraintCategory::UseAsStatic
                            } else {
                                ConstraintCategory::UseAsConst
                            }
                        } else {
                            ConstraintCategory::Return(ReturnConstraint::Normal)
                        }
                    }
                    Some(l)
                        if matches!(
                            self.body.local_decls[l].local_info(),
                            LocalInfo::AggregateTemp
                        ) =>
                    {
                        ConstraintCategory::Usage
                    }
                    Some(l) if !self.body.local_decls[l].is_user_variable() => {
                        ConstraintCategory::Boring
                    }
                    _ => ConstraintCategory::Assignment,
                };
                debug!(
                    "assignment category: {:?} {:?}",
                    category,
                    place.as_local().map(|l| &self.body.local_decls[l])
                );

                let place_ty = place.ty(self.body, tcx).ty;
                debug!(?place_ty);
                let place_ty = self.normalize(place_ty, location);
                debug!("place_ty normalized: {:?}", place_ty);
                let rv_ty = rv.ty(self.body, tcx);
                debug!(?rv_ty);
                let rv_ty = self.normalize(rv_ty, location);
                debug!("normalized rv_ty: {:?}", rv_ty);
                if let Err(terr) =
                    self.sub_types(rv_ty, place_ty, location.to_locations(), category)
                {
                    span_mirbug!(
                        self,
                        stmt,
                        "bad assignment ({:?} = {:?}): {:?}",
                        place_ty,
                        rv_ty,
                        terr
                    );
                }

                if let Some(annotation_index) = self.rvalue_user_ty(rv) {
                    if let Err(terr) = self.relate_type_and_user_type(
                        rv_ty,
                        ty::Invariant,
                        &UserTypeProjection { base: annotation_index, projs: vec![] },
                        location.to_locations(),
                        ConstraintCategory::TypeAnnotation(AnnotationSource::GenericArg),
                    ) {
                        let annotation = &self.user_type_annotations[annotation_index];
                        span_mirbug!(
                            self,
                            stmt,
                            "bad user type on rvalue ({:?} = {:?}): {:?}",
                            annotation,
                            rv_ty,
                            terr
                        );
                    }
                }

                if !self.unsized_feature_enabled() {
                    let trait_ref = ty::TraitRef::new(
                        tcx,
                        tcx.require_lang_item(LangItem::Sized, Some(self.last_span)),
                        [place_ty],
                    );
                    self.prove_trait_ref(
                        trait_ref,
                        location.to_locations(),
                        ConstraintCategory::SizedBound,
                    );
                }
            }
            StatementKind::AscribeUserType(box (place, projection), variance) => {
                let place_ty = place.ty(self.body, tcx).ty;
                if let Err(terr) = self.relate_type_and_user_type(
                    place_ty,
                    *variance,
                    projection,
                    Locations::All(stmt.source_info.span),
                    ConstraintCategory::TypeAnnotation(AnnotationSource::Ascription),
                ) {
                    let annotation = &self.user_type_annotations[projection.base];
                    span_mirbug!(
                        self,
                        stmt,
                        "bad type assert ({:?} <: {:?} with projections {:?}): {:?}",
                        place_ty,
                        annotation,
                        projection.projs,
                        terr
                    );
                }
            }
            StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(..))
            | StatementKind::FakeRead(..)
            | StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Retag { .. }
            | StatementKind::Coverage(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::PlaceMention(..)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Nop => {}
            StatementKind::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(..))
            | StatementKind::Deinit(..)
            | StatementKind::SetDiscriminant { .. } => {
                bug!("Statement not allowed in this MIR phase")
            }
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_terminator(&mut self, term: &Terminator<'tcx>, term_location: Location) {
        self.super_terminator(term, term_location);
        let tcx = self.tcx();
        debug!("terminator kind: {:?}", term.kind);
        match &term.kind {
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindResume
            | TerminatorKind::UnwindTerminate(_)
            | TerminatorKind::Return
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::InlineAsm { .. } => {
                // no checks needed for these
            }

            TerminatorKind::SwitchInt { discr, .. } => {
                let switch_ty = discr.ty(self.body, tcx);
                if !switch_ty.is_integral() && !switch_ty.is_char() && !switch_ty.is_bool() {
                    span_mirbug!(self, term, "bad SwitchInt discr ty {:?}", switch_ty);
                }
                // FIXME: check the values
            }
            TerminatorKind::Call { func, args, .. }
            | TerminatorKind::TailCall { func, args, .. } => {
                let call_source = match term.kind {
                    TerminatorKind::Call { call_source, .. } => call_source,
                    TerminatorKind::TailCall { .. } => CallSource::Normal,
                    _ => unreachable!(),
                };

                let func_ty = func.ty(self.body, tcx);
                debug!("func_ty.kind: {:?}", func_ty.kind());

                let sig = match func_ty.kind() {
                    ty::FnDef(..) | ty::FnPtr(..) => func_ty.fn_sig(tcx),
                    _ => {
                        span_mirbug!(self, term, "call to non-function {:?}", func_ty);
                        return;
                    }
                };
                let (unnormalized_sig, map) = tcx.instantiate_bound_regions(sig, |br| {
                    use crate::renumber::RegionCtxt;

                    let region_ctxt_fn = || {
                        let reg_info = match br.kind {
                            ty::BoundRegionKind::Anon => sym::anon,
                            ty::BoundRegionKind::Named(_, name) => name,
                            ty::BoundRegionKind::ClosureEnv => sym::env,
                        };

                        RegionCtxt::LateBound(reg_info)
                    };

                    self.infcx.next_region_var(
                        BoundRegion(
                            term.source_info.span,
                            br.kind,
                            BoundRegionConversionTime::FnCall,
                        ),
                        region_ctxt_fn,
                    )
                });
                debug!(?unnormalized_sig);
                // IMPORTANT: We have to prove well formed for the function signature before
                // we normalize it, as otherwise types like `<&'a &'b () as Trait>::Assoc`
                // get normalized away, causing us to ignore the `'b: 'a` bound used by the function.
                //
                // Normalization results in a well formed type if the input is well formed, so we
                // don't have to check it twice.
                //
                // See #91068 for an example.
                self.prove_predicates(
                    unnormalized_sig.inputs_and_output.iter().map(|ty| {
                        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                            ty.into(),
                        )))
                    }),
                    term_location.to_locations(),
                    ConstraintCategory::Boring,
                );

                let sig = self.deeply_normalize(unnormalized_sig, term_location);
                // HACK(#114936): `WF(sig)` does not imply `WF(normalized(sig))`
                // with built-in `Fn` implementations, since the impl may not be
                // well-formed itself.
                if sig != unnormalized_sig {
                    self.prove_predicates(
                        sig.inputs_and_output.iter().map(|ty| {
                            ty::Binder::dummy(ty::PredicateKind::Clause(
                                ty::ClauseKind::WellFormed(ty.into()),
                            ))
                        }),
                        term_location.to_locations(),
                        ConstraintCategory::Boring,
                    );
                }

                if let TerminatorKind::Call { destination, target, .. } = term.kind {
                    self.check_call_dest(term, &sig, destination, target, term_location);
                }

                // The ordinary liveness rules will ensure that all
                // regions in the type of the callee are live here. We
                // then further constrain the late-bound regions that
                // were instantiated at the call site to be live as
                // well. The resulting is that all the input (and
                // output) types in the signature must be live, since
                // all the inputs that fed into it were live.
                for &late_bound_region in map.values() {
                    let region_vid = self.universal_regions.to_region_vid(late_bound_region);
                    self.constraints.liveness_constraints.add_location(region_vid, term_location);
                }

                self.check_call_inputs(term, func, &sig, args, term_location, call_source);
            }
            TerminatorKind::Assert { cond, msg, .. } => {
                let cond_ty = cond.ty(self.body, tcx);
                if cond_ty != tcx.types.bool {
                    span_mirbug!(self, term, "bad Assert ({:?}, not bool", cond_ty);
                }

                if let AssertKind::BoundsCheck { len, index } = &**msg {
                    if len.ty(self.body, tcx) != tcx.types.usize {
                        span_mirbug!(self, len, "bounds-check length non-usize {:?}", len)
                    }
                    if index.ty(self.body, tcx) != tcx.types.usize {
                        span_mirbug!(self, index, "bounds-check index non-usize {:?}", index)
                    }
                }
            }
            TerminatorKind::Yield { value, resume_arg, .. } => {
                match self.body.yield_ty() {
                    None => span_mirbug!(self, term, "yield in non-coroutine"),
                    Some(ty) => {
                        let value_ty = value.ty(self.body, tcx);
                        if let Err(terr) = self.sub_types(
                            value_ty,
                            ty,
                            term_location.to_locations(),
                            ConstraintCategory::Yield,
                        ) {
                            span_mirbug!(
                                self,
                                term,
                                "type of yield value is {:?}, but the yield type is {:?}: {:?}",
                                value_ty,
                                ty,
                                terr
                            );
                        }
                    }
                }

                match self.body.resume_ty() {
                    None => span_mirbug!(self, term, "yield in non-coroutine"),
                    Some(ty) => {
                        let resume_ty = resume_arg.ty(self.body, tcx);
                        if let Err(terr) = self.sub_types(
                            ty,
                            resume_ty.ty,
                            term_location.to_locations(),
                            ConstraintCategory::Yield,
                        ) {
                            span_mirbug!(
                                self,
                                term,
                                "type of resume place is {:?}, but the resume type is {:?}: {:?}",
                                resume_ty,
                                ty,
                                terr
                            );
                        }
                    }
                }
            }
        }
    }

    fn visit_local_decl(&mut self, local: Local, local_decl: &LocalDecl<'tcx>) {
        self.super_local_decl(local, local_decl);

        for user_ty in
            local_decl.user_ty.as_deref().into_iter().flat_map(UserTypeProjections::projections)
        {
            let span = self.user_type_annotations[user_ty.base].span;

            let ty = if local_decl.is_nonref_binding() {
                local_decl.ty
            } else if let &ty::Ref(_, rty, _) = local_decl.ty.kind() {
                // If we have a binding of the form `let ref x: T = ..`
                // then remove the outermost reference so we can check the
                // type annotation for the remaining type.
                rty
            } else {
                bug!("{:?} with ref binding has wrong type {}", local, local_decl.ty);
            };

            if let Err(terr) = self.relate_type_and_user_type(
                ty,
                ty::Invariant,
                user_ty,
                Locations::All(span),
                ConstraintCategory::TypeAnnotation(AnnotationSource::Declaration),
            ) {
                span_mirbug!(
                    self,
                    local,
                    "bad user type on variable {:?}: {:?} != {:?} ({:?})",
                    local,
                    local_decl.ty,
                    local_decl.user_ty,
                    terr,
                );
            }
        }

        // When `unsized_fn_params` or `unsized_locals` is enabled, only function calls
        // and nullary ops are checked in `check_call_dest`.
        if !self.unsized_feature_enabled() {
            match self.body.local_kind(local) {
                LocalKind::ReturnPointer | LocalKind::Arg => {
                    // return values of normal functions are required to be
                    // sized by typeck, but return values of ADT constructors are
                    // not because we don't include a `Self: Sized` bounds on them.
                    //
                    // Unbound parts of arguments were never required to be Sized
                    // - maybe we should make that a warning.
                    return;
                }
                LocalKind::Temp => {
                    let span = local_decl.source_info.span;
                    let ty = local_decl.ty;
                    self.ensure_place_sized(ty, span);
                }
            }
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        let tcx = self.tcx();
        let span = self.body.source_info(location).span;
        match rvalue {
            Rvalue::Aggregate(ak, ops) => self.check_aggregate_rvalue(rvalue, ak, ops, location),

            Rvalue::Repeat(operand, len) => {
                let array_ty = rvalue.ty(self.body.local_decls(), tcx);
                self.prove_predicate(
                    ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(array_ty.into())),
                    Locations::Single(location),
                    ConstraintCategory::Boring,
                );

                // If the length cannot be evaluated we must assume that the length can be larger
                // than 1.
                // If the length is larger than 1, the repeat expression will need to copy the
                // element, so we require the `Copy` trait.
                if len.try_to_target_usize(tcx).is_none_or(|len| len > 1) {
                    match operand {
                        Operand::Copy(..) | Operand::Constant(..) => {
                            // These are always okay: direct use of a const, or a value that can
                            // evidently be copied.
                        }
                        Operand::Move(place) => {
                            // Make sure that repeated elements implement `Copy`.
                            let ty = place.ty(self.body, tcx).ty;
                            let trait_ref = ty::TraitRef::new(
                                tcx,
                                tcx.require_lang_item(LangItem::Copy, Some(span)),
                                [ty],
                            );

                            self.prove_trait_ref(
                                trait_ref,
                                Locations::Single(location),
                                ConstraintCategory::CopyBound,
                            );
                        }
                    }
                }
            }

            &Rvalue::NullaryOp(NullOp::SizeOf | NullOp::AlignOf, ty) => {
                let trait_ref = ty::TraitRef::new(
                    tcx,
                    tcx.require_lang_item(LangItem::Sized, Some(span)),
                    [ty],
                );

                self.prove_trait_ref(
                    trait_ref,
                    location.to_locations(),
                    ConstraintCategory::SizedBound,
                );
            }
            &Rvalue::NullaryOp(NullOp::ContractChecks, _) => {}
            &Rvalue::NullaryOp(NullOp::UbChecks, _) => {}

            Rvalue::ShallowInitBox(_operand, ty) => {
                let trait_ref = ty::TraitRef::new(
                    tcx,
                    tcx.require_lang_item(LangItem::Sized, Some(span)),
                    [*ty],
                );

                self.prove_trait_ref(
                    trait_ref,
                    location.to_locations(),
                    ConstraintCategory::SizedBound,
                );
            }

            Rvalue::Cast(cast_kind, op, ty) => {
                match *cast_kind {
                    CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer, coercion_source) => {
                        let is_implicit_coercion = coercion_source == CoercionSource::Implicit;
                        let src_ty = op.ty(self.body, tcx);
                        let mut src_sig = src_ty.fn_sig(tcx);
                        if let ty::FnDef(def_id, _) = src_ty.kind()
                            && let ty::FnPtr(_, target_hdr) = *ty.kind()
                            && tcx.codegen_fn_attrs(def_id).safe_target_features
                            && target_hdr.safety.is_safe()
                            && let Some(safe_sig) = tcx.adjust_target_feature_sig(
                                *def_id,
                                src_sig,
                                self.body.source.def_id(),
                            )
                        {
                            src_sig = safe_sig;
                        }

                        // HACK: This shouldn't be necessary... We can remove this when we actually
                        // get binders with where clauses, then elaborate implied bounds into that
                        // binder, and implement a higher-ranked subtyping algorithm that actually
                        // respects these implied bounds.
                        //
                        // This protects against the case where we are casting from a higher-ranked
                        // fn item to a non-higher-ranked fn pointer, where the cast throws away
                        // implied bounds that would've needed to be checked at the call site. This
                        // only works when we're casting to a non-higher-ranked fn ptr, since
                        // placeholders in the target signature could have untracked implied
                        // bounds, resulting in incorrect errors.
                        //
                        // We check that this signature is WF before subtyping the signature with
                        // the target fn sig.
                        if src_sig.has_bound_regions()
                            && let ty::FnPtr(target_fn_tys, target_hdr) = *ty.kind()
                            && let target_sig = target_fn_tys.with(target_hdr)
                            && let Some(target_sig) = target_sig.no_bound_vars()
                        {
                            let src_sig = self.infcx.instantiate_binder_with_fresh_vars(
                                span,
                                BoundRegionConversionTime::HigherRankedType,
                                src_sig,
                            );
                            let src_ty = Ty::new_fn_ptr(self.tcx(), ty::Binder::dummy(src_sig));
                            self.prove_predicate(
                                ty::ClauseKind::WellFormed(src_ty.into()),
                                location.to_locations(),
                                ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                            );

                            let src_ty = self.normalize(src_ty, location);
                            if let Err(terr) = self.sub_types(
                                src_ty,
                                *ty,
                                location.to_locations(),
                                ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                            ) {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "equating {:?} with {:?} yields {:?}",
                                    target_sig,
                                    src_sig,
                                    terr
                                );
                            };
                        }

                        let src_ty = Ty::new_fn_ptr(tcx, src_sig);
                        // HACK: We want to assert that the signature of the source fn is
                        // well-formed, because we don't enforce that via the WF of FnDef
                        // types normally. This should be removed when we improve the tracking
                        // of implied bounds of fn signatures.
                        self.prove_predicate(
                            ty::ClauseKind::WellFormed(src_ty.into()),
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        );

                        // The type that we see in the fcx is like
                        // `foo::<'a, 'b>`, where `foo` is the path to a
                        // function definition. When we extract the
                        // signature, it comes from the `fn_sig` query,
                        // and hence may contain unnormalized results.
                        let src_ty = self.normalize(src_ty, location);
                        if let Err(terr) = self.sub_types(
                            src_ty,
                            *ty,
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        ) {
                            span_mirbug!(
                                self,
                                rvalue,
                                "equating {:?} with {:?} yields {:?}",
                                src_ty,
                                ty,
                                terr
                            );
                        }
                    }

                    CastKind::PointerCoercion(
                        PointerCoercion::ClosureFnPointer(safety),
                        coercion_source,
                    ) => {
                        let sig = match op.ty(self.body, tcx).kind() {
                            ty::Closure(_, args) => args.as_closure().sig(),
                            _ => bug!(),
                        };
                        let ty_fn_ptr_from =
                            Ty::new_fn_ptr(tcx, tcx.signature_unclosure(sig, safety));

                        let is_implicit_coercion = coercion_source == CoercionSource::Implicit;
                        if let Err(terr) = self.sub_types(
                            ty_fn_ptr_from,
                            *ty,
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        ) {
                            span_mirbug!(
                                self,
                                rvalue,
                                "equating {:?} with {:?} yields {:?}",
                                ty_fn_ptr_from,
                                ty,
                                terr
                            );
                        }
                    }

                    CastKind::PointerCoercion(
                        PointerCoercion::UnsafeFnPointer,
                        coercion_source,
                    ) => {
                        let fn_sig = op.ty(self.body, tcx).fn_sig(tcx);

                        // The type that we see in the fcx is like
                        // `foo::<'a, 'b>`, where `foo` is the path to a
                        // function definition. When we extract the
                        // signature, it comes from the `fn_sig` query,
                        // and hence may contain unnormalized results.
                        let fn_sig = self.normalize(fn_sig, location);

                        let ty_fn_ptr_from = tcx.safe_to_unsafe_fn_ty(fn_sig);

                        let is_implicit_coercion = coercion_source == CoercionSource::Implicit;
                        if let Err(terr) = self.sub_types(
                            ty_fn_ptr_from,
                            *ty,
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        ) {
                            span_mirbug!(
                                self,
                                rvalue,
                                "equating {:?} with {:?} yields {:?}",
                                ty_fn_ptr_from,
                                ty,
                                terr
                            );
                        }
                    }

                    CastKind::PointerCoercion(PointerCoercion::Unsize, coercion_source) => {
                        let &ty = ty;
                        let trait_ref = ty::TraitRef::new(
                            tcx,
                            tcx.require_lang_item(LangItem::CoerceUnsized, Some(span)),
                            [op.ty(self.body, tcx), ty],
                        );

                        let is_implicit_coercion = coercion_source == CoercionSource::Implicit;
                        let unsize_to = fold_regions(tcx, ty, |r, _| {
                            if let ty::ReVar(_) = r.kind() { tcx.lifetimes.re_erased } else { r }
                        });
                        self.prove_trait_ref(
                            trait_ref,
                            location.to_locations(),
                            ConstraintCategory::Cast {
                                is_implicit_coercion,
                                unsize_to: Some(unsize_to),
                            },
                        );
                    }

                    CastKind::PointerCoercion(PointerCoercion::DynStar, coercion_source) => {
                        // get the constraints from the target type (`dyn* Clone`)
                        //
                        // apply them to prove that the source type `Foo` implements `Clone` etc
                        let (existential_predicates, region) = match ty.kind() {
                            Dynamic(predicates, region, ty::DynStar) => (predicates, region),
                            _ => panic!("Invalid dyn* cast_ty"),
                        };

                        let self_ty = op.ty(self.body, tcx);

                        let is_implicit_coercion = coercion_source == CoercionSource::Implicit;
                        self.prove_predicates(
                            existential_predicates
                                .iter()
                                .map(|predicate| predicate.with_self_ty(tcx, self_ty)),
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        );

                        let outlives_predicate = tcx.mk_predicate(Binder::dummy(
                            ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(
                                ty::OutlivesPredicate(self_ty, *region),
                            )),
                        ));
                        self.prove_predicate(
                            outlives_predicate,
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        );
                    }

                    CastKind::PointerCoercion(
                        PointerCoercion::MutToConstPointer,
                        coercion_source,
                    ) => {
                        let ty::RawPtr(ty_from, hir::Mutability::Mut) =
                            op.ty(self.body, tcx).kind()
                        else {
                            span_mirbug!(self, rvalue, "unexpected base type for cast {:?}", ty,);
                            return;
                        };
                        let ty::RawPtr(ty_to, hir::Mutability::Not) = ty.kind() else {
                            span_mirbug!(self, rvalue, "unexpected target type for cast {:?}", ty,);
                            return;
                        };
                        let is_implicit_coercion = coercion_source == CoercionSource::Implicit;
                        if let Err(terr) = self.sub_types(
                            *ty_from,
                            *ty_to,
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        ) {
                            span_mirbug!(
                                self,
                                rvalue,
                                "relating {:?} with {:?} yields {:?}",
                                ty_from,
                                ty_to,
                                terr
                            );
                        }
                    }

                    CastKind::PointerCoercion(PointerCoercion::ArrayToPointer, coercion_source) => {
                        let ty_from = op.ty(self.body, tcx);

                        let opt_ty_elem_mut = match ty_from.kind() {
                            ty::RawPtr(array_ty, array_mut) => match array_ty.kind() {
                                ty::Array(ty_elem, _) => Some((ty_elem, *array_mut)),
                                _ => None,
                            },
                            _ => None,
                        };

                        let Some((ty_elem, ty_mut)) = opt_ty_elem_mut else {
                            span_mirbug!(
                                self,
                                rvalue,
                                "ArrayToPointer cast from unexpected type {:?}",
                                ty_from,
                            );
                            return;
                        };

                        let (ty_to, ty_to_mut) = match ty.kind() {
                            ty::RawPtr(ty_to, ty_to_mut) => (ty_to, *ty_to_mut),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "ArrayToPointer cast to unexpected type {:?}",
                                    ty,
                                );
                                return;
                            }
                        };

                        if ty_to_mut.is_mut() && ty_mut.is_not() {
                            span_mirbug!(
                                self,
                                rvalue,
                                "ArrayToPointer cast from const {:?} to mut {:?}",
                                ty,
                                ty_to
                            );
                            return;
                        }

                        let is_implicit_coercion = coercion_source == CoercionSource::Implicit;
                        if let Err(terr) = self.sub_types(
                            *ty_elem,
                            *ty_to,
                            location.to_locations(),
                            ConstraintCategory::Cast { is_implicit_coercion, unsize_to: None },
                        ) {
                            span_mirbug!(
                                self,
                                rvalue,
                                "relating {:?} with {:?} yields {:?}",
                                ty_elem,
                                ty_to,
                                terr
                            )
                        }
                    }

                    CastKind::PointerExposeProvenance => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Ptr(_) | CastTy::FnPtr), Some(CastTy::Int(_))) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid PointerExposeProvenance cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }

                    CastKind::PointerWithExposedProvenance => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Int(_)), Some(CastTy::Ptr(_))) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid PointerWithExposedProvenance cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::IntToInt => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Int(_)), Some(CastTy::Int(_))) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid IntToInt cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::IntToFloat => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Int(_)), Some(CastTy::Float)) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid IntToFloat cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::FloatToInt => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Float), Some(CastTy::Int(_))) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid FloatToInt cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::FloatToFloat => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Float), Some(CastTy::Float)) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid FloatToFloat cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::FnPtrToPtr => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::FnPtr), Some(CastTy::Ptr(_))) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid FnPtrToPtr cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::PtrToPtr => {
                        let ty_from = op.ty(self.body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Ptr(src)), Some(CastTy::Ptr(dst))) => {
                                let src_tail = self.struct_tail(src.ty, location);
                                let dst_tail = self.struct_tail(dst.ty, location);

                                // This checks (lifetime part of) vtable validity for pointer casts,
                                // which is irrelevant when there are aren't principal traits on
                                // both sides (aka only auto traits).
                                //
                                // Note that other checks (such as denying `dyn Send` -> `dyn
                                // Debug`) are in `rustc_hir_typeck`.
                                if let ty::Dynamic(src_tty, _src_lt, ty::Dyn) = *src_tail.kind()
                                    && let ty::Dynamic(dst_tty, dst_lt, ty::Dyn) = *dst_tail.kind()
                                    && src_tty.principal().is_some()
                                    && dst_tty.principal().is_some()
                                {
                                    // Remove auto traits.
                                    // Auto trait checks are handled in `rustc_hir_typeck` as FCW.
                                    let src_obj = Ty::new_dynamic(
                                        tcx,
                                        tcx.mk_poly_existential_predicates(
                                            &src_tty.without_auto_traits().collect::<Vec<_>>(),
                                        ),
                                        // FIXME: Once we disallow casting `*const dyn Trait + 'short`
                                        // to `*const dyn Trait + 'long`, then this can just be `src_lt`.
                                        dst_lt,
                                        ty::Dyn,
                                    );
                                    let dst_obj = Ty::new_dynamic(
                                        tcx,
                                        tcx.mk_poly_existential_predicates(
                                            &dst_tty.without_auto_traits().collect::<Vec<_>>(),
                                        ),
                                        dst_lt,
                                        ty::Dyn,
                                    );

                                    debug!(?src_tty, ?dst_tty, ?src_obj, ?dst_obj);

                                    self.sub_types(
                                        src_obj,
                                        dst_obj,
                                        location.to_locations(),
                                        ConstraintCategory::Cast {
                                            is_implicit_coercion: false,
                                            unsize_to: None,
                                        },
                                    )
                                    .unwrap();
                                }
                            }
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid PtrToPtr cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::Transmute => {
                        span_mirbug!(
                            self,
                            rvalue,
                            "Unexpected CastKind::Transmute, which is not permitted in Analysis MIR",
                        );
                    }
                }
            }

            Rvalue::Ref(region, _borrow_kind, borrowed_place) => {
                self.add_reborrow_constraint(location, *region, borrowed_place);
            }

            Rvalue::BinaryOp(
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge,
                box (left, right),
            ) => {
                let ty_left = left.ty(self.body, tcx);
                match ty_left.kind() {
                    // Types with regions are comparable if they have a common super-type.
                    ty::RawPtr(_, _) | ty::FnPtr(..) => {
                        let ty_right = right.ty(self.body, tcx);
                        let common_ty =
                            self.infcx.next_ty_var(self.body.source_info(location).span);
                        self.sub_types(
                            ty_left,
                            common_ty,
                            location.to_locations(),
                            ConstraintCategory::CallArgument(None),
                        )
                        .unwrap_or_else(|err| {
                            bug!("Could not equate type variable with {:?}: {:?}", ty_left, err)
                        });
                        if let Err(terr) = self.sub_types(
                            ty_right,
                            common_ty,
                            location.to_locations(),
                            ConstraintCategory::CallArgument(None),
                        ) {
                            span_mirbug!(
                                self,
                                rvalue,
                                "unexpected comparison types {:?} and {:?} yields {:?}",
                                ty_left,
                                ty_right,
                                terr
                            )
                        }
                    }
                    // For types with no regions we can just check that the
                    // both operands have the same type.
                    ty::Int(_) | ty::Uint(_) | ty::Bool | ty::Char | ty::Float(_)
                        if ty_left == right.ty(self.body, tcx) => {}
                    // Other types are compared by trait methods, not by
                    // `Rvalue::BinaryOp`.
                    _ => span_mirbug!(
                        self,
                        rvalue,
                        "unexpected comparison types {:?} and {:?}",
                        ty_left,
                        right.ty(self.body, tcx)
                    ),
                }
            }

            Rvalue::WrapUnsafeBinder(op, ty) => {
                let operand_ty = op.ty(self.body, self.tcx());
                let ty::UnsafeBinder(binder_ty) = *ty.kind() else {
                    unreachable!();
                };
                let expected_ty = self.infcx.instantiate_binder_with_fresh_vars(
                    self.body().source_info(location).span,
                    BoundRegionConversionTime::HigherRankedType,
                    binder_ty.into(),
                );
                self.sub_types(
                    operand_ty,
                    expected_ty,
                    location.to_locations(),
                    ConstraintCategory::Boring,
                )
                .unwrap();
            }

            Rvalue::Use(_)
            | Rvalue::UnaryOp(_, _)
            | Rvalue::CopyForDeref(_)
            | Rvalue::BinaryOp(..)
            | Rvalue::RawPtr(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Len(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(NullOp::OffsetOf(..), _) => {}
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_operand(&mut self, op: &Operand<'tcx>, location: Location) {
        self.super_operand(op, location);
        if let Operand::Constant(constant) = op {
            let maybe_uneval = match constant.const_ {
                Const::Val(..) | Const::Ty(_, _) => None,
                Const::Unevaluated(uv, _) => Some(uv),
            };

            if let Some(uv) = maybe_uneval {
                if uv.promoted.is_none() {
                    let tcx = self.tcx();
                    let def_id = uv.def;
                    if tcx.def_kind(def_id) == DefKind::InlineConst {
                        let def_id = def_id.expect_local();
                        let predicates = self.prove_closure_bounds(
                            tcx,
                            def_id,
                            uv.args,
                            location.to_locations(),
                        );
                        self.normalize_and_prove_instantiated_predicates(
                            def_id.to_def_id(),
                            predicates,
                            location.to_locations(),
                        );
                    }
                }
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_const_operand(&mut self, constant: &ConstOperand<'tcx>, location: Location) {
        self.super_const_operand(constant, location);
        let ty = constant.const_.ty();

        self.infcx.tcx.for_each_free_region(&ty, |live_region| {
            let live_region_vid = self.universal_regions.to_region_vid(live_region);
            self.constraints.liveness_constraints.add_location(live_region_vid, location);
        });

        let locations = location.to_locations();
        if let Some(annotation_index) = constant.user_ty {
            if let Err(terr) = self.relate_type_and_user_type(
                constant.const_.ty(),
                ty::Invariant,
                &UserTypeProjection { base: annotation_index, projs: vec![] },
                locations,
                ConstraintCategory::TypeAnnotation(AnnotationSource::GenericArg),
            ) {
                let annotation = &self.user_type_annotations[annotation_index];
                span_mirbug!(
                    self,
                    constant,
                    "bad constant user type {:?} vs {:?}: {:?}",
                    annotation,
                    constant.const_.ty(),
                    terr,
                );
            }
        } else {
            let tcx = self.tcx();
            let maybe_uneval = match constant.const_ {
                Const::Ty(_, ct) => match ct.kind() {
                    ty::ConstKind::Unevaluated(uv) => {
                        Some(UnevaluatedConst { def: uv.def, args: uv.args, promoted: None })
                    }
                    _ => None,
                },
                Const::Unevaluated(uv, _) => Some(uv),
                _ => None,
            };

            if let Some(uv) = maybe_uneval {
                if let Some(promoted) = uv.promoted {
                    let promoted_body = &self.promoted[promoted];
                    self.check_promoted(promoted_body, location);
                    let promoted_ty = promoted_body.return_ty();
                    if let Err(terr) =
                        self.eq_types(ty, promoted_ty, locations, ConstraintCategory::Boring)
                    {
                        span_mirbug!(
                            self,
                            promoted,
                            "bad promoted type ({:?}: {:?}): {:?}",
                            ty,
                            promoted_ty,
                            terr
                        );
                    };
                } else {
                    self.ascribe_user_type(
                        constant.const_.ty(),
                        ty::UserType::new(ty::UserTypeKind::TypeOf(
                            uv.def,
                            UserArgs { args: uv.args, user_self_ty: None },
                        )),
                        locations.span(self.body),
                    );
                }
            } else if let Some(static_def_id) = constant.check_static_ptr(tcx) {
                let unnormalized_ty = tcx.type_of(static_def_id).instantiate_identity();
                let normalized_ty = self.normalize(unnormalized_ty, locations);
                let literal_ty = constant.const_.ty().builtin_deref(true).unwrap();

                if let Err(terr) =
                    self.eq_types(literal_ty, normalized_ty, locations, ConstraintCategory::Boring)
                {
                    span_mirbug!(self, constant, "bad static type {:?} ({:?})", constant, terr);
                }
            } else if let Const::Ty(_, ct) = constant.const_
                && let ty::ConstKind::Param(p) = ct.kind()
            {
                let body_def_id = self.universal_regions.defining_ty.def_id();
                let const_param = tcx.generics_of(body_def_id).const_param(p, tcx);
                self.ascribe_user_type(
                    constant.const_.ty(),
                    ty::UserType::new(ty::UserTypeKind::TypeOf(
                        const_param.def_id,
                        UserArgs {
                            args: self.universal_regions.defining_ty.args(),
                            user_self_ty: None,
                        },
                    )),
                    locations.span(self.body),
                );
            }

            if let ty::FnDef(def_id, args) = *constant.const_.ty().kind() {
                let instantiated_predicates = tcx.predicates_of(def_id).instantiate(tcx, args);
                self.normalize_and_prove_instantiated_predicates(
                    def_id,
                    instantiated_predicates,
                    locations,
                );

                assert!(!matches!(
                    tcx.impl_of_method(def_id).map(|imp| tcx.def_kind(imp)),
                    Some(DefKind::Impl { of_trait: true })
                ));
                self.prove_predicates(
                    args.types().map(|ty| ty::ClauseKind::WellFormed(ty.into())),
                    locations,
                    ConstraintCategory::Boring,
                );
            }
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.super_place(place, context, location);
        let tcx = self.tcx();
        let place_ty = place.ty(self.body, tcx);
        if let PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy) = context {
            let trait_ref = ty::TraitRef::new(
                tcx,
                tcx.require_lang_item(LangItem::Copy, Some(self.last_span)),
                [place_ty.ty],
            );

            // To have a `Copy` operand, the type `T` of the
            // value must be `Copy`. Note that we prove that `T: Copy`,
            // rather than using the `is_copy_modulo_regions`
            // test. This is important because
            // `is_copy_modulo_regions` ignores the resulting region
            // obligations and assumes they pass. This can result in
            // bounds from `Copy` impls being unsoundly ignored (e.g.,
            // #29149). Note that we decide to use `Copy` before knowing
            // whether the bounds fully apply: in effect, the rule is
            // that if a value of some type could implement `Copy`, then
            // it must.
            self.prove_trait_ref(trait_ref, location.to_locations(), ConstraintCategory::CopyBound);
        }
    }

    fn visit_projection_elem(
        &mut self,
        place: PlaceRef<'tcx>,
        elem: PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) {
        let tcx = self.tcx();
        let base_ty = place.ty(self.body(), tcx);
        match elem {
            // All these projections don't add any constraints, so there's nothing to
            // do here. We check their invariants in the MIR validator after all.
            ProjectionElem::Deref
            | ProjectionElem::Index(_)
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Downcast(..) => {}
            ProjectionElem::Field(field, fty) => {
                let fty = self.normalize(fty, location);
                let ty = base_ty.field_ty(tcx, field);
                let ty = self.normalize(ty, location);
                debug!(?fty, ?ty);

                if let Err(terr) = self.relate_types(
                    ty,
                    context.ambient_variance(),
                    fty,
                    location.to_locations(),
                    ConstraintCategory::Boring,
                ) {
                    span_mirbug!(self, place, "bad field access ({:?}: {:?}): {:?}", ty, fty, terr);
                }
            }
            ProjectionElem::OpaqueCast(ty) => {
                let ty = self.normalize(ty, location);
                self.relate_types(
                    ty,
                    context.ambient_variance(),
                    base_ty.ty,
                    location.to_locations(),
                    ConstraintCategory::TypeAnnotation(AnnotationSource::OpaqueCast),
                )
                .unwrap();
            }
            ProjectionElem::UnwrapUnsafeBinder(ty) => {
                let ty::UnsafeBinder(binder_ty) = *base_ty.ty.kind() else {
                    unreachable!();
                };
                let found_ty = self.infcx.instantiate_binder_with_fresh_vars(
                    self.body.source_info(location).span,
                    BoundRegionConversionTime::HigherRankedType,
                    binder_ty.into(),
                );
                self.relate_types(
                    ty,
                    context.ambient_variance(),
                    found_ty,
                    location.to_locations(),
                    ConstraintCategory::Boring,
                )
                .unwrap();
            }
            ProjectionElem::Subtype(_) => {
                bug!("ProjectionElem::Subtype shouldn't exist in borrowck")
            }
        }
    }
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn check_call_dest(
        &mut self,
        term: &Terminator<'tcx>,
        sig: &ty::FnSig<'tcx>,
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        term_location: Location,
    ) {
        let tcx = self.tcx();
        match target {
            Some(_) => {
                let dest_ty = destination.ty(self.body, tcx).ty;
                let dest_ty = self.normalize(dest_ty, term_location);
                let category = match destination.as_local() {
                    Some(RETURN_PLACE) => {
                        if let DefiningTy::Const(def_id, _) | DefiningTy::InlineConst(def_id, _) =
                            self.universal_regions.defining_ty
                        {
                            if tcx.is_static(def_id) {
                                ConstraintCategory::UseAsStatic
                            } else {
                                ConstraintCategory::UseAsConst
                            }
                        } else {
                            ConstraintCategory::Return(ReturnConstraint::Normal)
                        }
                    }
                    Some(l) if !self.body.local_decls[l].is_user_variable() => {
                        ConstraintCategory::Boring
                    }
                    // The return type of a call is interesting for diagnostics.
                    _ => ConstraintCategory::Assignment,
                };

                let locations = term_location.to_locations();

                if let Err(terr) = self.sub_types(sig.output(), dest_ty, locations, category) {
                    span_mirbug!(
                        self,
                        term,
                        "call dest mismatch ({:?} <- {:?}): {:?}",
                        dest_ty,
                        sig.output(),
                        terr
                    );
                }

                // When `unsized_fn_params` and `unsized_locals` are both not enabled,
                // this check is done at `check_local`.
                if self.unsized_feature_enabled() {
                    let span = term.source_info.span;
                    self.ensure_place_sized(dest_ty, span);
                }
            }
            None => {
                // The signature in this call can reference region variables,
                // so erase them before calling a query.
                let output_ty = self.tcx().erase_regions(sig.output());
                if !output_ty.is_privately_uninhabited(
                    self.tcx(),
                    self.infcx.typing_env(self.infcx.param_env),
                ) {
                    span_mirbug!(self, term, "call to converging function {:?} w/o dest", sig);
                }
            }
        }
    }

    #[instrument(level = "debug", skip(self, term, func, term_location, call_source))]
    fn check_call_inputs(
        &mut self,
        term: &Terminator<'tcx>,
        func: &Operand<'tcx>,
        sig: &ty::FnSig<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        term_location: Location,
        call_source: CallSource,
    ) {
        if args.len() < sig.inputs().len() || (args.len() > sig.inputs().len() && !sig.c_variadic) {
            span_mirbug!(self, term, "call to {:?} with wrong # of args", sig);
        }

        let func_ty = func.ty(self.body, self.infcx.tcx);
        if let ty::FnDef(def_id, _) = *func_ty.kind() {
            // Some of the SIMD intrinsics are special: they need a particular argument to be a
            // constant. (Eventually this should use const-generics, but those are not up for the
            // task yet: https://github.com/rust-lang/rust/issues/85229.)
            if let Some(name @ (sym::simd_shuffle | sym::simd_insert | sym::simd_extract)) =
                self.tcx().intrinsic(def_id).map(|i| i.name)
            {
                let idx = match name {
                    sym::simd_shuffle => 2,
                    _ => 1,
                };
                if !matches!(args[idx], Spanned { node: Operand::Constant(_), .. }) {
                    self.tcx().dcx().emit_err(SimdIntrinsicArgConst {
                        span: term.source_info.span,
                        arg: idx + 1,
                        intrinsic: name.to_string(),
                    });
                }
            }
        }
        debug!(?func_ty);

        for (n, (fn_arg, op_arg)) in iter::zip(sig.inputs(), args).enumerate() {
            let op_arg_ty = op_arg.node.ty(self.body, self.tcx());

            let op_arg_ty = self.normalize(op_arg_ty, term_location);
            let category = if call_source.from_hir_call() {
                ConstraintCategory::CallArgument(Some(self.infcx.tcx.erase_regions(func_ty)))
            } else {
                ConstraintCategory::Boring
            };
            if let Err(terr) =
                self.sub_types(op_arg_ty, *fn_arg, term_location.to_locations(), category)
            {
                span_mirbug!(
                    self,
                    term,
                    "bad arg #{:?} ({:?} <- {:?}): {:?}",
                    n,
                    fn_arg,
                    op_arg_ty,
                    terr
                );
            }
        }
    }

    fn check_iscleanup(&mut self, block_data: &BasicBlockData<'tcx>) {
        let is_cleanup = block_data.is_cleanup;
        match block_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                self.assert_iscleanup(block_data, target, is_cleanup)
            }
            TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets.all_targets() {
                    self.assert_iscleanup(block_data, *target, is_cleanup);
                }
            }
            TerminatorKind::UnwindResume => {
                if !is_cleanup {
                    span_mirbug!(self, block_data, "resume on non-cleanup block!")
                }
            }
            TerminatorKind::UnwindTerminate(_) => {
                if !is_cleanup {
                    span_mirbug!(self, block_data, "terminate on non-cleanup block!")
                }
            }
            TerminatorKind::Return => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "return on cleanup block")
                }
            }
            TerminatorKind::TailCall { .. } => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "tailcall on cleanup block")
                }
            }
            TerminatorKind::CoroutineDrop { .. } => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "coroutine_drop in cleanup block")
                }
            }
            TerminatorKind::Yield { resume, drop, .. } => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "yield in cleanup block")
                }
                self.assert_iscleanup(block_data, resume, is_cleanup);
                if let Some(drop) = drop {
                    self.assert_iscleanup(block_data, drop, is_cleanup);
                }
            }
            TerminatorKind::Unreachable => {}
            TerminatorKind::Drop { target, unwind, .. }
            | TerminatorKind::Assert { target, unwind, .. } => {
                self.assert_iscleanup(block_data, target, is_cleanup);
                self.assert_iscleanup_unwind(block_data, unwind, is_cleanup);
            }
            TerminatorKind::Call { ref target, unwind, .. } => {
                if let &Some(target) = target {
                    self.assert_iscleanup(block_data, target, is_cleanup);
                }
                self.assert_iscleanup_unwind(block_data, unwind, is_cleanup);
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                self.assert_iscleanup(block_data, real_target, is_cleanup);
                self.assert_iscleanup(block_data, imaginary_target, is_cleanup);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                self.assert_iscleanup(block_data, real_target, is_cleanup);
                self.assert_iscleanup_unwind(block_data, unwind, is_cleanup);
            }
            TerminatorKind::InlineAsm { ref targets, unwind, .. } => {
                for &target in targets {
                    self.assert_iscleanup(block_data, target, is_cleanup);
                }
                self.assert_iscleanup_unwind(block_data, unwind, is_cleanup);
            }
        }
    }

    fn assert_iscleanup(&mut self, ctxt: &dyn fmt::Debug, bb: BasicBlock, iscleanuppad: bool) {
        if self.body[bb].is_cleanup != iscleanuppad {
            span_mirbug!(self, ctxt, "cleanuppad mismatch: {:?} should be {:?}", bb, iscleanuppad);
        }
    }

    fn assert_iscleanup_unwind(
        &mut self,
        ctxt: &dyn fmt::Debug,
        unwind: UnwindAction,
        is_cleanup: bool,
    ) {
        match unwind {
            UnwindAction::Cleanup(unwind) => {
                if is_cleanup {
                    span_mirbug!(self, ctxt, "unwind on cleanup block")
                }
                self.assert_iscleanup(ctxt, unwind, true);
            }
            UnwindAction::Continue => {
                if is_cleanup {
                    span_mirbug!(self, ctxt, "unwind on cleanup block")
                }
            }
            UnwindAction::Unreachable | UnwindAction::Terminate(_) => (),
        }
    }

    fn ensure_place_sized(&mut self, ty: Ty<'tcx>, span: Span) {
        let tcx = self.tcx();

        // Erase the regions from `ty` to get a global type. The
        // `Sized` bound in no way depends on precise regions, so this
        // shouldn't affect `is_sized`.
        let erased_ty = tcx.erase_regions(ty);
        // FIXME(#132279): Using `Ty::is_sized` causes us to incorrectly handle opaques here.
        if !erased_ty.is_sized(tcx, self.infcx.typing_env(self.infcx.param_env)) {
            // in current MIR construction, all non-control-flow rvalue
            // expressions evaluate through `as_temp` or `into` a return
            // slot or local, so to find all unsized rvalues it is enough
            // to check all temps, return slots and locals.
            if self.reported_errors.replace((ty, span)).is_none() {
                // While this is located in `nll::typeck` this error is not
                // an NLL error, it's a required check to prevent creation
                // of unsized rvalues in a call expression.
                self.tcx().dcx().emit_err(MoveUnsized { ty, span });
            }
        }
    }

    fn aggregate_field_ty(
        &mut self,
        ak: &AggregateKind<'tcx>,
        field_index: FieldIdx,
        location: Location,
    ) -> Result<Ty<'tcx>, FieldAccessError> {
        let tcx = self.tcx();

        match *ak {
            AggregateKind::Adt(adt_did, variant_index, args, _, active_field_index) => {
                let def = tcx.adt_def(adt_did);
                let variant = &def.variant(variant_index);
                let adj_field_index = active_field_index.unwrap_or(field_index);
                if let Some(field) = variant.fields.get(adj_field_index) {
                    Ok(self.normalize(field.ty(tcx, args), location))
                } else {
                    Err(FieldAccessError::OutOfRange { field_count: variant.fields.len() })
                }
            }
            AggregateKind::Closure(_, args) => {
                match args.as_closure().upvar_tys().get(field_index.as_usize()) {
                    Some(ty) => Ok(*ty),
                    None => Err(FieldAccessError::OutOfRange {
                        field_count: args.as_closure().upvar_tys().len(),
                    }),
                }
            }
            AggregateKind::Coroutine(_, args) => {
                // It doesn't make sense to look at a field beyond the prefix;
                // these require a variant index, and are not initialized in
                // aggregate rvalues.
                match args.as_coroutine().prefix_tys().get(field_index.as_usize()) {
                    Some(ty) => Ok(*ty),
                    None => Err(FieldAccessError::OutOfRange {
                        field_count: args.as_coroutine().prefix_tys().len(),
                    }),
                }
            }
            AggregateKind::CoroutineClosure(_, args) => {
                match args.as_coroutine_closure().upvar_tys().get(field_index.as_usize()) {
                    Some(ty) => Ok(*ty),
                    None => Err(FieldAccessError::OutOfRange {
                        field_count: args.as_coroutine_closure().upvar_tys().len(),
                    }),
                }
            }
            AggregateKind::Array(ty) => Ok(ty),
            AggregateKind::Tuple | AggregateKind::RawPtr(..) => {
                unreachable!("This should have been covered in check_rvalues");
            }
        }
    }

    /// If this rvalue supports a user-given type annotation, then
    /// extract and return it. This represents the final type of the
    /// rvalue and will be unified with the inferred type.
    fn rvalue_user_ty(&self, rvalue: &Rvalue<'tcx>) -> Option<UserTypeAnnotationIndex> {
        match rvalue {
            Rvalue::Use(_)
            | Rvalue::ThreadLocalRef(_)
            | Rvalue::Repeat(..)
            | Rvalue::Ref(..)
            | Rvalue::RawPtr(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::CopyForDeref(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::WrapUnsafeBinder(..) => None,

            Rvalue::Aggregate(aggregate, _) => match **aggregate {
                AggregateKind::Adt(_, _, _, user_ty, _) => user_ty,
                AggregateKind::Array(_) => None,
                AggregateKind::Tuple => None,
                AggregateKind::Closure(_, _) => None,
                AggregateKind::Coroutine(_, _) => None,
                AggregateKind::CoroutineClosure(_, _) => None,
                AggregateKind::RawPtr(_, _) => None,
            },
        }
    }

    fn check_aggregate_rvalue(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        aggregate_kind: &AggregateKind<'tcx>,
        operands: &IndexSlice<FieldIdx, Operand<'tcx>>,
        location: Location,
    ) {
        let tcx = self.tcx();

        self.prove_aggregate_predicates(aggregate_kind, location);

        if *aggregate_kind == AggregateKind::Tuple {
            // tuple rvalue field type is always the type of the op. Nothing to check here.
            return;
        }

        if let AggregateKind::RawPtr(..) = aggregate_kind {
            bug!("RawPtr should only be in runtime MIR");
        }

        for (i, operand) in operands.iter_enumerated() {
            let field_ty = match self.aggregate_field_ty(aggregate_kind, i, location) {
                Ok(field_ty) => field_ty,
                Err(FieldAccessError::OutOfRange { field_count }) => {
                    span_mirbug!(
                        self,
                        rvalue,
                        "accessed field #{} but variant only has {}",
                        i.as_u32(),
                        field_count,
                    );
                    continue;
                }
            };
            let operand_ty = operand.ty(self.body, tcx);
            let operand_ty = self.normalize(operand_ty, location);

            if let Err(terr) = self.sub_types(
                operand_ty,
                field_ty,
                location.to_locations(),
                ConstraintCategory::Boring,
            ) {
                span_mirbug!(
                    self,
                    rvalue,
                    "{:?} is not a subtype of {:?}: {:?}",
                    operand_ty,
                    field_ty,
                    terr
                );
            }
        }
    }

    /// Adds the constraints that arise from a borrow expression `&'a P` at the location `L`.
    ///
    /// # Parameters
    ///
    /// - `location`: the location `L` where the borrow expression occurs
    /// - `borrow_region`: the region `'a` associated with the borrow
    /// - `borrowed_place`: the place `P` being borrowed
    fn add_reborrow_constraint(
        &mut self,
        location: Location,
        borrow_region: ty::Region<'tcx>,
        borrowed_place: &Place<'tcx>,
    ) {
        // These constraints are only meaningful during borrowck:
        let Self { borrow_set, location_table, polonius_facts, constraints, .. } = self;

        // In Polonius mode, we also push a `loan_issued_at` fact
        // linking the loan to the region (in some cases, though,
        // there is no loan associated with this borrow expression --
        // that occurs when we are borrowing an unsafe place, for
        // example).
        if let Some(polonius_facts) = polonius_facts {
            let _prof_timer = self.infcx.tcx.prof.generic_activity("polonius_fact_generation");
            if let Some(borrow_index) = borrow_set.get_index_of(&location) {
                let region_vid = borrow_region.as_var();
                polonius_facts.loan_issued_at.push((
                    region_vid.into(),
                    borrow_index,
                    location_table.mid_index(location),
                ));
            }
        }

        // If we are reborrowing the referent of another reference, we
        // need to add outlives relationships. In a case like `&mut
        // *p`, where the `p` has type `&'b mut Foo`, for example, we
        // need to ensure that `'b: 'a`.

        debug!(
            "add_reborrow_constraint({:?}, {:?}, {:?})",
            location, borrow_region, borrowed_place
        );

        let tcx = self.infcx.tcx;
        let def = self.body.source.def_id().expect_local();
        let upvars = tcx.closure_captures(def);
        let field =
            path_utils::is_upvar_field_projection(tcx, upvars, borrowed_place.as_ref(), self.body);
        let category = if let Some(field) = field {
            ConstraintCategory::ClosureUpvar(field)
        } else {
            ConstraintCategory::Boring
        };

        for (base, elem) in borrowed_place.as_ref().iter_projections().rev() {
            debug!("add_reborrow_constraint - iteration {:?}", elem);

            match elem {
                ProjectionElem::Deref => {
                    let base_ty = base.ty(self.body, tcx).ty;

                    debug!("add_reborrow_constraint - base_ty = {:?}", base_ty);
                    match base_ty.kind() {
                        ty::Ref(ref_region, _, mutbl) => {
                            constraints.outlives_constraints.push(OutlivesConstraint {
                                sup: ref_region.as_var(),
                                sub: borrow_region.as_var(),
                                locations: location.to_locations(),
                                span: location.to_locations().span(self.body),
                                category,
                                variance_info: ty::VarianceDiagInfo::default(),
                                from_closure: false,
                            });

                            match mutbl {
                                hir::Mutability::Not => {
                                    // Immutable reference. We don't need the base
                                    // to be valid for the entire lifetime of
                                    // the borrow.
                                    break;
                                }
                                hir::Mutability::Mut => {
                                    // Mutable reference. We *do* need the base
                                    // to be valid, because after the base becomes
                                    // invalid, someone else can use our mutable deref.

                                    // This is in order to make the following function
                                    // illegal:
                                    // ```
                                    // fn unsafe_deref<'a, 'b>(x: &'a &'b mut T) -> &'b mut T {
                                    //     &mut *x
                                    // }
                                    // ```
                                    //
                                    // As otherwise you could clone `&mut T` using the
                                    // following function:
                                    // ```
                                    // fn bad(x: &mut T) -> (&mut T, &mut T) {
                                    //     let my_clone = unsafe_deref(&'a x);
                                    //     ENDREGION 'a;
                                    //     (my_clone, x)
                                    // }
                                    // ```
                                }
                            }
                        }
                        ty::RawPtr(..) => {
                            // deref of raw pointer, guaranteed to be valid
                            break;
                        }
                        ty::Adt(def, _) if def.is_box() => {
                            // deref of `Box`, need the base to be valid - propagate
                        }
                        _ => bug!("unexpected deref ty {:?} in {:?}", base_ty, borrowed_place),
                    }
                }
                ProjectionElem::Field(..)
                | ProjectionElem::Downcast(..)
                | ProjectionElem::OpaqueCast(..)
                | ProjectionElem::Index(..)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::UnwrapUnsafeBinder(_) => {
                    // other field access
                }
                ProjectionElem::Subtype(_) => {
                    bug!("ProjectionElem::Subtype shouldn't exist in borrowck")
                }
            }
        }
    }

    fn prove_aggregate_predicates(
        &mut self,
        aggregate_kind: &AggregateKind<'tcx>,
        location: Location,
    ) {
        let tcx = self.tcx();

        debug!(
            "prove_aggregate_predicates(aggregate_kind={:?}, location={:?})",
            aggregate_kind, location
        );

        let (def_id, instantiated_predicates) = match *aggregate_kind {
            AggregateKind::Adt(adt_did, _, args, _, _) => {
                (adt_did, tcx.predicates_of(adt_did).instantiate(tcx, args))
            }

            // For closures, we have some **extra requirements** we
            // have to check. In particular, in their upvars and
            // signatures, closures often reference various regions
            // from the surrounding function -- we call those the
            // closure's free regions. When we borrow-check (and hence
            // region-check) closures, we may find that the closure
            // requires certain relationships between those free
            // regions. However, because those free regions refer to
            // portions of the CFG of their caller, the closure is not
            // in a position to verify those relationships. In that
            // case, the requirements get "propagated" to us, and so
            // we have to solve them here where we instantiate the
            // closure.
            //
            // Despite the opacity of the previous paragraph, this is
            // actually relatively easy to understand in terms of the
            // desugaring. A closure gets desugared to a struct, and
            // these extra requirements are basically like where
            // clauses on the struct.
            AggregateKind::Closure(def_id, args)
            | AggregateKind::CoroutineClosure(def_id, args)
            | AggregateKind::Coroutine(def_id, args) => (
                def_id,
                self.prove_closure_bounds(
                    tcx,
                    def_id.expect_local(),
                    args,
                    location.to_locations(),
                ),
            ),

            AggregateKind::Array(_) | AggregateKind::Tuple | AggregateKind::RawPtr(..) => {
                (CRATE_DEF_ID.to_def_id(), ty::InstantiatedPredicates::empty())
            }
        };

        self.normalize_and_prove_instantiated_predicates(
            def_id,
            instantiated_predicates,
            location.to_locations(),
        );
    }

    fn prove_closure_bounds(
        &mut self,
        tcx: TyCtxt<'tcx>,
        def_id: LocalDefId,
        args: GenericArgsRef<'tcx>,
        locations: Locations,
    ) -> ty::InstantiatedPredicates<'tcx> {
        if let Some(closure_requirements) = &self.root_cx.closure_requirements(def_id) {
            constraint_conversion::ConstraintConversion::new(
                self.infcx,
                self.universal_regions,
                &self.region_bound_pairs,
                self.implicit_region_bound,
                self.infcx.param_env,
                &self.known_type_outlives_obligations,
                locations,
                self.body.span,             // irrelevant; will be overridden.
                ConstraintCategory::Boring, // same as above.
                self.constraints,
            )
            .apply_closure_requirements(closure_requirements, def_id, args);
        }

        // Now equate closure args to regions inherited from `typeck_root_def_id`. Fixes #98589.
        let typeck_root_def_id = tcx.typeck_root_def_id(self.body.source.def_id());
        let typeck_root_args = ty::GenericArgs::identity_for_item(tcx, typeck_root_def_id);

        let parent_args = match tcx.def_kind(def_id) {
            // We don't want to dispatch on 3 different kind of closures here, so take
            // advantage of the fact that the `parent_args` is the same length as the
            // `typeck_root_args`.
            DefKind::Closure => {
                // FIXME(async_closures): It may be useful to add a debug assert here
                // to actually call `type_of` and check the `parent_args` are the same
                // length as the `typeck_root_args`.
                &args[..typeck_root_args.len()]
            }
            DefKind::InlineConst => args.as_inline_const().parent_args(),
            other => bug!("unexpected item {:?}", other),
        };
        let parent_args = tcx.mk_args(parent_args);

        assert_eq!(typeck_root_args.len(), parent_args.len());
        if let Err(_) = self.eq_args(
            typeck_root_args,
            parent_args,
            locations,
            ConstraintCategory::BoringNoLocation,
        ) {
            span_mirbug!(
                self,
                def_id,
                "could not relate closure to parent {:?} != {:?}",
                typeck_root_args,
                parent_args
            );
        }

        tcx.predicates_of(def_id).instantiate(tcx, args)
    }
}

trait NormalizeLocation: fmt::Debug + Copy {
    fn to_locations(self) -> Locations;
}

impl NormalizeLocation for Locations {
    fn to_locations(self) -> Locations {
        self
    }
}

impl NormalizeLocation for Location {
    fn to_locations(self) -> Locations {
        Locations::Single(self)
    }
}

/// Runs `infcx.instantiate_opaque_types`. Unlike other `TypeOp`s,
/// this is not canonicalized - it directly affects the main `InferCtxt`
/// that we use during MIR borrowchecking.
#[derive(Debug)]
pub(super) struct InstantiateOpaqueType<'tcx> {
    pub base_universe: Option<ty::UniverseIndex>,
    pub region_constraints: Option<RegionConstraintData<'tcx>>,
    pub obligations: PredicateObligations<'tcx>,
}

impl<'tcx> TypeOp<'tcx> for InstantiateOpaqueType<'tcx> {
    type Output = ();
    /// We use this type itself to store the information used
    /// when reporting errors. Since this is not a query, we don't
    /// re-run anything during error reporting - we just use the information
    /// we saved to help extract an error from the already-existing region
    /// constraints in our `InferCtxt`
    type ErrorInfo = InstantiateOpaqueType<'tcx>;

    fn fully_perform(
        mut self,
        infcx: &InferCtxt<'tcx>,
        span: Span,
    ) -> Result<TypeOpOutput<'tcx, Self>, ErrorGuaranteed> {
        let (mut output, region_constraints) = scrape_region_constraints(
            infcx,
            |ocx| {
                ocx.register_obligations(self.obligations.clone());
                Ok(())
            },
            "InstantiateOpaqueType",
            span,
        )?;
        self.region_constraints = Some(region_constraints);
        output.error_info = Some(self);
        Ok(output)
    }
}
