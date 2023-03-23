#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
//! This pass type-checks the MIR to ensure it is not broken.

use std::rc::Rc;
use std::{fmt, iter, mem};

use either::Either;

use hir::OpaqueTyOrigin;
use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::lang_items::LangItem;
use rustc_index::vec::{Idx, IndexVec};
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::infer::outlives::env::RegionBoundPairs;
use rustc_infer::infer::region_constraints::RegionConstraintData;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{
    InferCtxt, InferOk, LateBoundRegion, LateBoundRegionConversionTime, NllRegionVariableOrigin,
};
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::*;
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::cast::CastTy;
use rustc_middle::ty::subst::{SubstsRef, UserSubsts};
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{
    self, Binder, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations, Dynamic,
    OpaqueHiddenType, OpaqueTypeKey, RegionVid, Ty, TyCtxt, UserType, UserTypeAnnotationIndex,
};
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::VariantIdx;
use rustc_trait_selection::traits::query::type_op::custom::scrape_region_constraints;
use rustc_trait_selection::traits::query::type_op::custom::CustomTypeOp;
use rustc_trait_selection::traits::query::type_op::{TypeOp, TypeOpOutput};
use rustc_trait_selection::traits::query::Fallible;
use rustc_trait_selection::traits::PredicateObligation;

use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::ResultsCursor;

use crate::session_diagnostics::MoveUnsized;
use crate::{
    borrow_set::BorrowSet,
    constraints::{OutlivesConstraint, OutlivesConstraintSet},
    diagnostics::UniverseInfo,
    facts::AllFacts,
    location::LocationTable,
    member_constraints::MemberConstraintSet,
    nll::ToRegionVid,
    path_utils,
    region_infer::values::{
        LivenessValues, PlaceholderIndex, PlaceholderIndices, RegionValueElements,
    },
    region_infer::TypeTest,
    type_check::free_region_relations::{CreateResult, UniversalRegionRelations},
    universal_regions::{DefiningTy, UniversalRegions},
    BorrowckInferCtxt, Upvar,
};

macro_rules! span_mirbug {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        $crate::type_check::mirbug(
            $context.tcx(),
            $context.last_span,
            &format!(
                "broken MIR in {:?} ({:?}): {}",
                $context.body().source.def_id(),
                $elem,
                format_args!($($message)*),
            ),
        )
    })
}

macro_rules! span_mirbug_and_err {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        {
            span_mirbug!($context, $elem, $($message)*);
            $context.error()
        }
    })
}

mod canonical;
mod constraint_conversion;
pub mod free_region_relations;
mod input_output;
pub(crate) mod liveness;
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
/// - `param_env` -- parameter environment to use for trait solving
/// - `body` -- MIR body to type-check
/// - `promoted` -- map of promoted constants within `body`
/// - `universal_regions` -- the universal regions from `body`s function signature
/// - `location_table` -- MIR location map of `body`
/// - `borrow_set` -- information about borrows occurring in `body`
/// - `all_facts` -- when using Polonius, this is the generated set of Polonius facts
/// - `flow_inits` -- results of a maybe-init dataflow analysis
/// - `move_data` -- move-data constructed when performing the maybe-init dataflow analysis
/// - `elements` -- MIR region map
pub(crate) fn type_check<'mir, 'tcx>(
    infcx: &BorrowckInferCtxt<'_, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body: &Body<'tcx>,
    promoted: &IndexVec<Promoted, Body<'tcx>>,
    universal_regions: &Rc<UniversalRegions<'tcx>>,
    location_table: &LocationTable,
    borrow_set: &BorrowSet<'tcx>,
    all_facts: &mut Option<AllFacts>,
    flow_inits: &mut ResultsCursor<'mir, 'tcx, MaybeInitializedPlaces<'mir, 'tcx>>,
    move_data: &MoveData<'tcx>,
    elements: &Rc<RegionValueElements>,
    upvars: &[Upvar<'tcx>],
    use_polonius: bool,
) -> MirTypeckResults<'tcx> {
    let implicit_region_bound = infcx.tcx.mk_re_var(universal_regions.fr_fn_body);
    let mut constraints = MirTypeckRegionConstraints {
        placeholder_indices: PlaceholderIndices::default(),
        placeholder_index_to_region: IndexVec::default(),
        liveness_constraints: LivenessValues::new(elements.clone()),
        outlives_constraints: OutlivesConstraintSet::default(),
        member_constraints: MemberConstraintSet::default(),
        type_tests: Vec::default(),
        universe_causes: FxIndexMap::default(),
    };

    let CreateResult {
        universal_region_relations,
        region_bound_pairs,
        normalized_inputs_and_output,
    } = free_region_relations::create(
        infcx,
        param_env,
        implicit_region_bound,
        universal_regions,
        &mut constraints,
    );

    debug!(?normalized_inputs_and_output);

    for u in ty::UniverseIndex::ROOT..=infcx.universe() {
        constraints.universe_causes.insert(u, UniverseInfo::other());
    }

    let mut borrowck_context = BorrowCheckContext {
        universal_regions,
        location_table,
        borrow_set,
        all_facts,
        constraints: &mut constraints,
        upvars,
    };

    let mut checker = TypeChecker::new(
        infcx,
        body,
        param_env,
        &region_bound_pairs,
        implicit_region_bound,
        &mut borrowck_context,
    );

    let errors_reported = {
        let mut verifier = TypeVerifier::new(&mut checker, promoted);
        verifier.visit_body(&body);
        verifier.errors_reported
    };

    if !errors_reported {
        // if verifier failed, don't do further checks to avoid ICEs
        checker.typeck_mir(body);
    }

    checker.equate_inputs_and_outputs(&body, universal_regions, &normalized_inputs_and_output);
    checker.check_signature_annotation(&body);

    liveness::generate(
        &mut checker,
        body,
        elements,
        flow_inits,
        move_data,
        location_table,
        use_polonius,
    );

    translate_outlives_facts(&mut checker);
    let opaque_type_values = infcx.take_opaque_types();

    let opaque_type_values = opaque_type_values
        .into_iter()
        .map(|(opaque_type_key, decl)| {
            checker
                .fully_perform_op(
                    Locations::All(body.span),
                    ConstraintCategory::OpaqueType,
                    CustomTypeOp::new(
                        |infcx| {
                            infcx.register_member_constraints(
                                param_env,
                                opaque_type_key,
                                decl.hidden_type.ty,
                                decl.hidden_type.span,
                            );
                            Ok(InferOk { value: (), obligations: vec![] })
                        },
                        || "opaque_type_map".to_string(),
                    ),
                )
                .unwrap();
            let mut hidden_type = infcx.resolve_vars_if_possible(decl.hidden_type);
            trace!("finalized opaque type {:?} to {:#?}", opaque_type_key, hidden_type.ty.kind());
            if hidden_type.has_non_region_infer() {
                let reported = infcx.tcx.sess.delay_span_bug(
                    decl.hidden_type.span,
                    &format!("could not resolve {:#?}", hidden_type.ty.kind()),
                );
                hidden_type.ty = infcx.tcx.ty_error(reported);
            }

            (opaque_type_key, (hidden_type, decl.origin))
        })
        .collect();

    MirTypeckResults { constraints, universal_region_relations, opaque_type_values }
}

fn translate_outlives_facts(typeck: &mut TypeChecker<'_, '_>) {
    let cx = &mut typeck.borrowck_context;
    if let Some(facts) = cx.all_facts {
        let _prof_timer = typeck.infcx.tcx.prof.generic_activity("polonius_fact_generation");
        let location_table = cx.location_table;
        facts.subset_base.extend(cx.constraints.outlives_constraints.outlives().iter().flat_map(
            |constraint: &OutlivesConstraint<'_>| {
                if let Some(from_location) = constraint.locations.from_location() {
                    Either::Left(iter::once((
                        constraint.sup,
                        constraint.sub,
                        location_table.mid_index(from_location),
                    )))
                } else {
                    Either::Right(
                        location_table
                            .all_points()
                            .map(move |location| (constraint.sup, constraint.sub, location)),
                    )
                }
            },
        ));
    }
}

#[track_caller]
fn mirbug(tcx: TyCtxt<'_>, span: Span, msg: &str) {
    // We sometimes see MIR failures (notably predicate failures) due to
    // the fact that we check rvalue sized predicates here. So use `delay_span_bug`
    // to avoid reporting bugs in those cases.
    tcx.sess.diagnostic().delay_span_bug(span, msg);
}

enum FieldAccessError {
    OutOfRange { field_count: usize },
}

/// Verifies that MIR types are sane to not crash further checks.
///
/// The sanitize_XYZ methods here take an MIR object and compute its
/// type, calling `span_mirbug` and returning an error type if there
/// is a problem.
struct TypeVerifier<'a, 'b, 'tcx> {
    cx: &'a mut TypeChecker<'b, 'tcx>,
    promoted: &'b IndexVec<Promoted, Body<'tcx>>,
    last_span: Span,
    errors_reported: bool,
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for TypeVerifier<'a, 'b, 'tcx> {
    fn visit_span(&mut self, span: Span) {
        if !span.is_dummy() {
            self.last_span = span;
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.sanitize_place(place, location, context);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        debug!(?constant, ?location, "visit_constant");

        self.super_constant(constant, location);
        let ty = self.sanitize_type(constant, constant.literal.ty());

        self.cx.infcx.tcx.for_each_free_region(&ty, |live_region| {
            let live_region_vid =
                self.cx.borrowck_context.universal_regions.to_region_vid(live_region);
            self.cx
                .borrowck_context
                .constraints
                .liveness_constraints
                .add_element(live_region_vid, location);
        });

        // HACK(compiler-errors): Constants that are gathered into Body.required_consts
        // have their locations erased...
        let locations = if location != Location::START {
            location.to_locations()
        } else {
            Locations::All(constant.span)
        };

        if let Some(annotation_index) = constant.user_ty {
            if let Err(terr) = self.cx.relate_type_and_user_type(
                constant.literal.ty(),
                ty::Variance::Invariant,
                &UserTypeProjection { base: annotation_index, projs: vec![] },
                locations,
                ConstraintCategory::Boring,
            ) {
                let annotation = &self.cx.user_type_annotations[annotation_index];
                span_mirbug!(
                    self,
                    constant,
                    "bad constant user type {:?} vs {:?}: {:?}",
                    annotation,
                    constant.literal.ty(),
                    terr,
                );
            }
        } else {
            let tcx = self.tcx();
            let maybe_uneval = match constant.literal {
                ConstantKind::Ty(ct) => match ct.kind() {
                    ty::ConstKind::Unevaluated(_) => {
                        bug!("should not encounter unevaluated ConstantKind::Ty here, got {:?}", ct)
                    }
                    _ => None,
                },
                ConstantKind::Unevaluated(uv, _) => Some(uv),
                _ => None,
            };

            if let Some(uv) = maybe_uneval {
                if let Some(promoted) = uv.promoted {
                    let check_err = |verifier: &mut TypeVerifier<'a, 'b, 'tcx>,
                                     promoted: &Body<'tcx>,
                                     ty,
                                     san_ty| {
                        if let Err(terr) =
                            verifier.cx.eq_types(ty, san_ty, locations, ConstraintCategory::Boring)
                        {
                            span_mirbug!(
                                verifier,
                                promoted,
                                "bad promoted type ({:?}: {:?}): {:?}",
                                ty,
                                san_ty,
                                terr
                            );
                        };
                    };

                    if !self.errors_reported {
                        let promoted_body = &self.promoted[promoted];
                        self.sanitize_promoted(promoted_body, location);

                        let promoted_ty = promoted_body.return_ty();
                        check_err(self, promoted_body, ty, promoted_ty);
                    }
                } else {
                    self.cx.ascribe_user_type(
                        constant.literal.ty(),
                        UserType::TypeOf(
                            uv.def.did,
                            UserSubsts { substs: uv.substs, user_self_ty: None },
                        ),
                        locations.span(&self.cx.body),
                    );
                }
            } else if let Some(static_def_id) = constant.check_static_ptr(tcx) {
                let unnormalized_ty = tcx.type_of(static_def_id).subst_identity();
                let normalized_ty = self.cx.normalize(unnormalized_ty, locations);
                let literal_ty = constant.literal.ty().builtin_deref(true).unwrap().ty;

                if let Err(terr) = self.cx.eq_types(
                    literal_ty,
                    normalized_ty,
                    locations,
                    ConstraintCategory::Boring,
                ) {
                    span_mirbug!(self, constant, "bad static type {:?} ({:?})", constant, terr);
                }
            }

            if let ty::FnDef(def_id, substs) = *constant.literal.ty().kind() {
                // const_trait_impl: use a non-const param env when checking that a FnDef type is well formed.
                // this is because the well-formedness of the function does not need to be proved to have `const`
                // impls for trait bounds.
                let instantiated_predicates = tcx.predicates_of(def_id).instantiate(tcx, substs);
                let prev = self.cx.param_env;
                self.cx.param_env = prev.without_const();
                self.cx.normalize_and_prove_instantiated_predicates(
                    def_id,
                    instantiated_predicates,
                    locations,
                );
                self.cx.param_env = prev;
            }
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        let rval_ty = rvalue.ty(self.body(), self.tcx());
        self.sanitize_type(rvalue, rval_ty);
    }

    fn visit_local_decl(&mut self, local: Local, local_decl: &LocalDecl<'tcx>) {
        self.super_local_decl(local, local_decl);
        self.sanitize_type(local_decl, local_decl.ty);

        if let Some(user_ty) = &local_decl.user_ty {
            for (user_ty, span) in user_ty.projections_and_spans() {
                let ty = if !local_decl.is_nonref_binding() {
                    // If we have a binding of the form `let ref x: T = ..`
                    // then remove the outermost reference so we can check the
                    // type annotation for the remaining type.
                    if let ty::Ref(_, rty, _) = local_decl.ty.kind() {
                        *rty
                    } else {
                        bug!("{:?} with ref binding has wrong type {}", local, local_decl.ty);
                    }
                } else {
                    local_decl.ty
                };

                if let Err(terr) = self.cx.relate_type_and_user_type(
                    ty,
                    ty::Variance::Invariant,
                    user_ty,
                    Locations::All(*span),
                    ConstraintCategory::TypeAnnotation,
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
        }
    }

    fn visit_body(&mut self, body: &Body<'tcx>) {
        self.sanitize_type(&"return type", body.return_ty());
        for local_decl in &body.local_decls {
            self.sanitize_type(local_decl, local_decl.ty);
        }
        if self.errors_reported {
            return;
        }
        self.super_body(body);
    }
}

impl<'a, 'b, 'tcx> TypeVerifier<'a, 'b, 'tcx> {
    fn new(
        cx: &'a mut TypeChecker<'b, 'tcx>,
        promoted: &'b IndexVec<Promoted, Body<'tcx>>,
    ) -> Self {
        TypeVerifier { promoted, last_span: cx.body.span, cx, errors_reported: false }
    }

    fn body(&self) -> &Body<'tcx> {
        self.cx.body
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.infcx.tcx
    }

    fn sanitize_type(&mut self, parent: &dyn fmt::Debug, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.has_escaping_bound_vars() || ty.references_error() {
            span_mirbug_and_err!(self, parent, "bad type {:?}", ty)
        } else {
            ty
        }
    }

    /// Checks that the types internal to the `place` match up with
    /// what would be expected.
    fn sanitize_place(
        &mut self,
        place: &Place<'tcx>,
        location: Location,
        context: PlaceContext,
    ) -> PlaceTy<'tcx> {
        debug!("sanitize_place: {:?}", place);

        let mut place_ty = PlaceTy::from_ty(self.body().local_decls[place.local].ty);

        for elem in place.projection.iter() {
            if place_ty.variant_index.is_none() {
                if let Err(guar) = place_ty.ty.error_reported() {
                    assert!(self.errors_reported);
                    return PlaceTy::from_ty(self.tcx().ty_error(guar));
                }
            }
            place_ty = self.sanitize_projection(place_ty, elem, place, location, context);
        }

        if let PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy) = context {
            let tcx = self.tcx();
            let trait_ref = tcx.at(self.last_span).mk_trait_ref(LangItem::Copy, [place_ty.ty]);

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
            self.cx.prove_trait_ref(
                trait_ref,
                location.to_locations(),
                ConstraintCategory::CopyBound,
            );
        }

        place_ty
    }

    fn sanitize_promoted(&mut self, promoted_body: &'b Body<'tcx>, location: Location) {
        // Determine the constraints from the promoted MIR by running the type
        // checker on the promoted MIR, then transfer the constraints back to
        // the main MIR, changing the locations to the provided location.

        let parent_body = mem::replace(&mut self.cx.body, promoted_body);

        // Use new sets of constraints and closure bounds so that we can
        // modify their locations.
        let all_facts = &mut None;
        let mut constraints = Default::default();
        let mut liveness_constraints =
            LivenessValues::new(Rc::new(RegionValueElements::new(&promoted_body)));
        // Don't try to add borrow_region facts for the promoted MIR

        let mut swap_constraints = |this: &mut Self| {
            mem::swap(this.cx.borrowck_context.all_facts, all_facts);
            mem::swap(
                &mut this.cx.borrowck_context.constraints.outlives_constraints,
                &mut constraints,
            );
            mem::swap(
                &mut this.cx.borrowck_context.constraints.liveness_constraints,
                &mut liveness_constraints,
            );
        };

        swap_constraints(self);

        self.visit_body(&promoted_body);

        if !self.errors_reported {
            // if verifier failed, don't do further checks to avoid ICEs
            self.cx.typeck_mir(promoted_body);
        }

        self.cx.body = parent_body;
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
            self.cx.borrowck_context.constraints.outlives_constraints.push(constraint)
        }
        for region in liveness_constraints.rows() {
            // If the region is live at at least one location in the promoted MIR,
            // then add a liveness constraint to the main MIR for this region
            // at the location provided as an argument to this method
            if liveness_constraints.get_elements(region).next().is_some() {
                self.cx
                    .borrowck_context
                    .constraints
                    .liveness_constraints
                    .add_element(region, location);
            }
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn sanitize_projection(
        &mut self,
        base: PlaceTy<'tcx>,
        pi: PlaceElem<'tcx>,
        place: &Place<'tcx>,
        location: Location,
        context: PlaceContext,
    ) -> PlaceTy<'tcx> {
        debug!("sanitize_projection: {:?} {:?} {:?}", base, pi, place);
        let tcx = self.tcx();
        let base_ty = base.ty;
        match pi {
            ProjectionElem::Deref => {
                let deref_ty = base_ty.builtin_deref(true);
                PlaceTy::from_ty(deref_ty.map(|t| t.ty).unwrap_or_else(|| {
                    span_mirbug_and_err!(self, place, "deref of non-pointer {:?}", base_ty)
                }))
            }
            ProjectionElem::Index(i) => {
                let index_ty = Place::from(i).ty(self.body(), tcx).ty;
                if index_ty != tcx.types.usize {
                    PlaceTy::from_ty(span_mirbug_and_err!(self, i, "index by non-usize {:?}", i))
                } else {
                    PlaceTy::from_ty(base_ty.builtin_index().unwrap_or_else(|| {
                        span_mirbug_and_err!(self, place, "index of non-array {:?}", base_ty)
                    }))
                }
            }
            ProjectionElem::ConstantIndex { .. } => {
                // consider verifying in-bounds
                PlaceTy::from_ty(base_ty.builtin_index().unwrap_or_else(|| {
                    span_mirbug_and_err!(self, place, "index of non-array {:?}", base_ty)
                }))
            }
            ProjectionElem::Subslice { from, to, from_end } => {
                PlaceTy::from_ty(match base_ty.kind() {
                    ty::Array(inner, _) => {
                        assert!(!from_end, "array subslices should not use from_end");
                        tcx.mk_array(*inner, to - from)
                    }
                    ty::Slice(..) => {
                        assert!(from_end, "slice subslices should use from_end");
                        base_ty
                    }
                    _ => span_mirbug_and_err!(self, place, "slice of non-array {:?}", base_ty),
                })
            }
            ProjectionElem::Downcast(maybe_name, index) => match base_ty.kind() {
                ty::Adt(adt_def, _substs) if adt_def.is_enum() => {
                    if index.as_usize() >= adt_def.variants().len() {
                        PlaceTy::from_ty(span_mirbug_and_err!(
                            self,
                            place,
                            "cast to variant #{:?} but enum only has {:?}",
                            index,
                            adt_def.variants().len()
                        ))
                    } else {
                        PlaceTy { ty: base_ty, variant_index: Some(index) }
                    }
                }
                // We do not need to handle generators here, because this runs
                // before the generator transform stage.
                _ => {
                    let ty = if let Some(name) = maybe_name {
                        span_mirbug_and_err!(
                            self,
                            place,
                            "can't downcast {:?} as {:?}",
                            base_ty,
                            name
                        )
                    } else {
                        span_mirbug_and_err!(self, place, "can't downcast {:?}", base_ty)
                    };
                    PlaceTy::from_ty(ty)
                }
            },
            ProjectionElem::Field(field, fty) => {
                let fty = self.sanitize_type(place, fty);
                let fty = self.cx.normalize(fty, location);
                match self.field_ty(place, base, field, location) {
                    Ok(ty) => {
                        let ty = self.cx.normalize(ty, location);
                        debug!(?fty, ?ty);

                        if let Err(terr) = self.cx.relate_types(
                            ty,
                            self.get_ambient_variance(context),
                            fty,
                            location.to_locations(),
                            ConstraintCategory::Boring,
                        ) {
                            span_mirbug!(
                                self,
                                place,
                                "bad field access ({:?}: {:?}): {:?}",
                                ty,
                                fty,
                                terr
                            );
                        }
                    }
                    Err(FieldAccessError::OutOfRange { field_count }) => span_mirbug!(
                        self,
                        place,
                        "accessed field #{} but variant only has {}",
                        field.index(),
                        field_count
                    ),
                }
                PlaceTy::from_ty(fty)
            }
            ProjectionElem::OpaqueCast(ty) => {
                let ty = self.sanitize_type(place, ty);
                let ty = self.cx.normalize(ty, location);
                self.cx
                    .relate_types(
                        ty,
                        self.get_ambient_variance(context),
                        base.ty,
                        location.to_locations(),
                        ConstraintCategory::TypeAnnotation,
                    )
                    .unwrap();
                PlaceTy::from_ty(ty)
            }
        }
    }

    fn error(&mut self) -> Ty<'tcx> {
        self.errors_reported = true;
        self.tcx().ty_error_misc()
    }

    fn get_ambient_variance(&self, context: PlaceContext) -> ty::Variance {
        use rustc_middle::mir::visit::NonMutatingUseContext::*;
        use rustc_middle::mir::visit::NonUseContext::*;

        match context {
            PlaceContext::MutatingUse(_) => ty::Invariant,
            PlaceContext::NonUse(StorageDead | StorageLive | PlaceMention | VarDebugInfo) => {
                ty::Invariant
            }
            PlaceContext::NonMutatingUse(
                Inspect | Copy | Move | SharedBorrow | ShallowBorrow | UniqueBorrow | AddressOf
                | Projection,
            ) => ty::Covariant,
            PlaceContext::NonUse(AscribeUserTy) => ty::Covariant,
        }
    }

    fn field_ty(
        &mut self,
        parent: &dyn fmt::Debug,
        base_ty: PlaceTy<'tcx>,
        field: Field,
        location: Location,
    ) -> Result<Ty<'tcx>, FieldAccessError> {
        let tcx = self.tcx();

        let (variant, substs) = match base_ty {
            PlaceTy { ty, variant_index: Some(variant_index) } => match *ty.kind() {
                ty::Adt(adt_def, substs) => (adt_def.variant(variant_index), substs),
                ty::Generator(def_id, substs, _) => {
                    let mut variants = substs.as_generator().state_tys(def_id, tcx);
                    let Some(mut variant) = variants.nth(variant_index.into()) else {
                        bug!(
                            "variant_index of generator out of range: {:?}/{:?}",
                            variant_index,
                            substs.as_generator().state_tys(def_id, tcx).count()
                        );
                    };
                    return match variant.nth(field.index()) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange { field_count: variant.count() }),
                    };
                }
                _ => bug!("can't have downcast of non-adt non-generator type"),
            },
            PlaceTy { ty, variant_index: None } => match *ty.kind() {
                ty::Adt(adt_def, substs) if !adt_def.is_enum() => {
                    (adt_def.variant(VariantIdx::new(0)), substs)
                }
                ty::Closure(_, substs) => {
                    return match substs
                        .as_closure()
                        .tupled_upvars_ty()
                        .tuple_fields()
                        .get(field.index())
                    {
                        Some(&ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.as_closure().upvar_tys().count(),
                        }),
                    };
                }
                ty::Generator(_, substs, _) => {
                    // Only prefix fields (upvars and current state) are
                    // accessible without a variant index.
                    return match substs.as_generator().prefix_tys().nth(field.index()) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.as_generator().prefix_tys().count(),
                        }),
                    };
                }
                ty::Tuple(tys) => {
                    return match tys.get(field.index()) {
                        Some(&ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange { field_count: tys.len() }),
                    };
                }
                _ => {
                    return Ok(span_mirbug_and_err!(
                        self,
                        parent,
                        "can't project out of {:?}",
                        base_ty
                    ));
                }
            },
        };

        if let Some(field) = variant.fields.get(field.index()) {
            Ok(self.cx.normalize(field.ty(tcx, substs), location))
        } else {
            Err(FieldAccessError::OutOfRange { field_count: variant.fields.len() })
        }
    }
}

/// The MIR type checker. Visits the MIR and enforces all the
/// constraints needed for it to be valid and well-typed. Along the
/// way, it accrues region constraints -- these can later be used by
/// NLL region checking.
struct TypeChecker<'a, 'tcx> {
    infcx: &'a BorrowckInferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    last_span: Span,
    body: &'a Body<'tcx>,
    /// User type annotations are shared between the main MIR and the MIR of
    /// all of the promoted items.
    user_type_annotations: &'a CanonicalUserTypeAnnotations<'tcx>,
    region_bound_pairs: &'a RegionBoundPairs<'tcx>,
    implicit_region_bound: ty::Region<'tcx>,
    reported_errors: FxIndexSet<(Ty<'tcx>, Span)>,
    borrowck_context: &'a mut BorrowCheckContext<'a, 'tcx>,
}

struct BorrowCheckContext<'a, 'tcx> {
    pub(crate) universal_regions: &'a UniversalRegions<'tcx>,
    location_table: &'a LocationTable,
    all_facts: &'a mut Option<AllFacts>,
    borrow_set: &'a BorrowSet<'tcx>,
    pub(crate) constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
    upvars: &'a [Upvar<'tcx>],
}

pub(crate) struct MirTypeckResults<'tcx> {
    pub(crate) constraints: MirTypeckRegionConstraints<'tcx>,
    pub(crate) universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
    pub(crate) opaque_type_values:
        FxIndexMap<OpaqueTypeKey<'tcx>, (OpaqueHiddenType<'tcx>, OpaqueTyOrigin)>,
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
    pub(crate) liveness_constraints: LivenessValues<RegionVid>,

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
    fn new(
        infcx: &'a BorrowckInferCtxt<'a, 'tcx>,
        body: &'a Body<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        region_bound_pairs: &'a RegionBoundPairs<'tcx>,
        implicit_region_bound: ty::Region<'tcx>,
        borrowck_context: &'a mut BorrowCheckContext<'a, 'tcx>,
    ) -> Self {
        let mut checker = Self {
            infcx,
            last_span: DUMMY_SP,
            body,
            user_type_annotations: &body.user_type_annotations,
            param_env,
            region_bound_pairs,
            implicit_region_bound,
            borrowck_context,
            reported_errors: Default::default(),
        };
        checker.check_user_type_annotations();
        checker
    }

    fn body(&self) -> &Body<'tcx> {
        self.body
    }

    fn unsized_feature_enabled(&self) -> bool {
        let features = self.tcx().features();
        features.unsized_locals || features.unsized_fn_params
    }

    /// Equate the inferred type and the annotated type for user type annotations
    #[instrument(skip(self), level = "debug")]
    fn check_user_type_annotations(&mut self) {
        debug!(?self.user_type_annotations);
        for user_annotation in self.user_type_annotations {
            let CanonicalUserTypeAnnotation { span, ref user_ty, inferred_ty } = *user_annotation;
            let annotation = self.instantiate_canonical_with_fresh_inference_vars(span, user_ty);
            self.ascribe_user_type(inferred_ty, annotation, span);
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
            self.borrowck_context.universal_regions,
            self.region_bound_pairs,
            self.implicit_region_bound,
            self.param_env,
            locations,
            locations.span(self.body),
            category,
            &mut self.borrowck_context.constraints,
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
    ) -> Fallible<()> {
        // Use this order of parameters because the sup type is usually the
        // "expected" type in diagnostics.
        self.relate_types(sup, ty::Variance::Contravariant, sub, locations, category)
    }

    #[instrument(skip(self, category), level = "debug")]
    fn eq_types(
        &mut self,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) -> Fallible<()> {
        self.relate_types(expected, ty::Variance::Invariant, found, locations, category)
    }

    #[instrument(skip(self), level = "debug")]
    fn relate_type_and_user_type(
        &mut self,
        a: Ty<'tcx>,
        v: ty::Variance,
        user_ty: &UserTypeProjection,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) -> Fallible<()> {
        let annotated_type = self.user_type_annotations[user_ty.base].inferred_ty;
        trace!(?annotated_type);
        let mut curr_projected_ty = PlaceTy::from_ty(annotated_type);

        let tcx = self.infcx.tcx;

        for proj in &user_ty.projs {
            if let ty::Alias(ty::Opaque, ..) = curr_projected_ty.ty.kind() {
                // There is nothing that we can compare here if we go through an opaque type.
                // We're always in its defining scope as we can otherwise not project through
                // it, so we're constraining it anyways.
                return Ok(());
            }
            let projected_ty = curr_projected_ty.projection_ty_core(
                tcx,
                self.param_env,
                proj,
                |this, field, ()| {
                    let ty = this.field_ty(tcx, field);
                    self.normalize(ty, locations)
                },
                |_, _| unreachable!(),
            );
            curr_projected_ty = projected_ty;
        }
        trace!(?curr_projected_ty);

        let ty = curr_projected_ty.ty;
        self.relate_types(ty, v.xform(ty::Variance::Contravariant), a, locations, category)?;

        Ok(())
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[instrument(skip(self, body, location), level = "debug")]
    fn check_stmt(&mut self, body: &Body<'tcx>, stmt: &Statement<'tcx>, location: Location) {
        let tcx = self.tcx();
        debug!("stmt kind: {:?}", stmt.kind);
        match &stmt.kind {
            StatementKind::Assign(box (place, rv)) => {
                // Assignments to temporaries are not "interesting";
                // they are not caused by the user, but rather artifacts
                // of lowering. Assignments to other sorts of places *are* interesting
                // though.
                let category = match place.as_local() {
                    Some(RETURN_PLACE) => {
                        let defining_ty = &self.borrowck_context.universal_regions.defining_ty;
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
                        if matches!(body.local_decls[l].local_info(), LocalInfo::AggregateTemp) =>
                    {
                        ConstraintCategory::Usage
                    }
                    Some(l) if !body.local_decls[l].is_user_variable() => {
                        ConstraintCategory::Boring
                    }
                    _ => ConstraintCategory::Assignment,
                };
                debug!(
                    "assignment category: {:?} {:?}",
                    category,
                    place.as_local().map(|l| &body.local_decls[l])
                );

                let place_ty = place.ty(body, tcx).ty;
                debug!(?place_ty);
                let place_ty = self.normalize(place_ty, location);
                debug!("place_ty normalized: {:?}", place_ty);
                let rv_ty = rv.ty(body, tcx);
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
                        ty::Variance::Invariant,
                        &UserTypeProjection { base: annotation_index, projs: vec![] },
                        location.to_locations(),
                        ConstraintCategory::Boring,
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

                self.check_rvalue(body, rv, location);
                if !self.unsized_feature_enabled() {
                    let trait_ref =
                        tcx.at(self.last_span).mk_trait_ref(LangItem::Sized, [place_ty]);
                    self.prove_trait_ref(
                        trait_ref,
                        location.to_locations(),
                        ConstraintCategory::SizedBound,
                    );
                }
            }
            StatementKind::AscribeUserType(box (place, projection), variance) => {
                let place_ty = place.ty(body, tcx).ty;
                if let Err(terr) = self.relate_type_and_user_type(
                    place_ty,
                    *variance,
                    projection,
                    Locations::All(stmt.source_info.span),
                    ConstraintCategory::TypeAnnotation,
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
            StatementKind::Intrinsic(box kind) => match kind {
                NonDivergingIntrinsic::Assume(op) => self.check_operand(op, location),
                NonDivergingIntrinsic::CopyNonOverlapping(..) => span_bug!(
                    stmt.source_info.span,
                    "Unexpected NonDivergingIntrinsic::CopyNonOverlapping, should only appear after lowering_intrinsics",
                ),
            },
            StatementKind::FakeRead(..)
            | StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Retag { .. }
            | StatementKind::Coverage(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::PlaceMention(..)
            | StatementKind::Nop => {}
            StatementKind::Deinit(..) | StatementKind::SetDiscriminant { .. } => {
                bug!("Statement not allowed in this MIR phase")
            }
        }
    }

    #[instrument(skip(self, body, term_location), level = "debug")]
    fn check_terminator(
        &mut self,
        body: &Body<'tcx>,
        term: &Terminator<'tcx>,
        term_location: Location,
    ) {
        let tcx = self.tcx();
        debug!("terminator kind: {:?}", term.kind);
        match &term.kind {
            TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::InlineAsm { .. } => {
                // no checks needed for these
            }

            TerminatorKind::SwitchInt { discr, .. } => {
                self.check_operand(discr, term_location);

                let switch_ty = discr.ty(body, tcx);
                if !switch_ty.is_integral() && !switch_ty.is_char() && !switch_ty.is_bool() {
                    span_mirbug!(self, term, "bad SwitchInt discr ty {:?}", switch_ty);
                }
                // FIXME: check the values
            }
            TerminatorKind::Call { func, args, destination, from_hir_call, target, .. } => {
                self.check_operand(func, term_location);
                for arg in args {
                    self.check_operand(arg, term_location);
                }

                let func_ty = func.ty(body, tcx);
                debug!("func_ty.kind: {:?}", func_ty.kind());

                let sig = match func_ty.kind() {
                    ty::FnDef(..) | ty::FnPtr(_) => func_ty.fn_sig(tcx),
                    _ => {
                        span_mirbug!(self, term, "call to non-function {:?}", func_ty);
                        return;
                    }
                };
                let (sig, map) = tcx.replace_late_bound_regions(sig, |br| {
                    use crate::renumber::{BoundRegionInfo, RegionCtxt};
                    use rustc_span::Symbol;

                    let region_ctxt_fn = || {
                        let reg_info = match br.kind {
                            ty::BoundRegionKind::BrAnon(_, Some(span)) => {
                                BoundRegionInfo::Span(span)
                            }
                            ty::BoundRegionKind::BrAnon(..) => {
                                BoundRegionInfo::Name(Symbol::intern("anon"))
                            }
                            ty::BoundRegionKind::BrNamed(_, name) => BoundRegionInfo::Name(name),
                            ty::BoundRegionKind::BrEnv => {
                                BoundRegionInfo::Name(Symbol::intern("env"))
                            }
                        };

                        RegionCtxt::LateBound(reg_info)
                    };

                    self.infcx.next_region_var(
                        LateBoundRegion(
                            term.source_info.span,
                            br.kind,
                            LateBoundRegionConversionTime::FnCall,
                        ),
                        region_ctxt_fn,
                    )
                });
                debug!(?sig);
                // IMPORTANT: We have to prove well formed for the function signature before
                // we normalize it, as otherwise types like `<&'a &'b () as Trait>::Assoc`
                // get normalized away, causing us to ignore the `'b: 'a` bound used by the function.
                //
                // Normalization results in a well formed type if the input is well formed, so we
                // don't have to check it twice.
                //
                // See #91068 for an example.
                self.prove_predicates(
                    sig.inputs_and_output
                        .iter()
                        .map(|ty| ty::Binder::dummy(ty::PredicateKind::WellFormed(ty.into()))),
                    term_location.to_locations(),
                    ConstraintCategory::Boring,
                );
                let sig = self.normalize(sig, term_location);
                self.check_call_dest(body, term, &sig, *destination, *target, term_location);

                // The ordinary liveness rules will ensure that all
                // regions in the type of the callee are live here. We
                // then further constrain the late-bound regions that
                // were instantiated at the call site to be live as
                // well. The resulting is that all the input (and
                // output) types in the signature must be live, since
                // all the inputs that fed into it were live.
                for &late_bound_region in map.values() {
                    let region_vid =
                        self.borrowck_context.universal_regions.to_region_vid(late_bound_region);
                    self.borrowck_context
                        .constraints
                        .liveness_constraints
                        .add_element(region_vid, term_location);
                }

                self.check_call_inputs(body, term, &sig, args, term_location, *from_hir_call);
            }
            TerminatorKind::Assert { cond, msg, .. } => {
                self.check_operand(cond, term_location);

                let cond_ty = cond.ty(body, tcx);
                if cond_ty != tcx.types.bool {
                    span_mirbug!(self, term, "bad Assert ({:?}, not bool", cond_ty);
                }

                if let AssertKind::BoundsCheck { len, index } = msg {
                    if len.ty(body, tcx) != tcx.types.usize {
                        span_mirbug!(self, len, "bounds-check length non-usize {:?}", len)
                    }
                    if index.ty(body, tcx) != tcx.types.usize {
                        span_mirbug!(self, index, "bounds-check index non-usize {:?}", index)
                    }
                }
            }
            TerminatorKind::Yield { value, .. } => {
                self.check_operand(value, term_location);

                let value_ty = value.ty(body, tcx);
                match body.yield_ty() {
                    None => span_mirbug!(self, term, "yield in non-generator"),
                    Some(ty) => {
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
            }
        }
    }

    fn check_call_dest(
        &mut self,
        body: &Body<'tcx>,
        term: &Terminator<'tcx>,
        sig: &ty::FnSig<'tcx>,
        destination: Place<'tcx>,
        target: Option<BasicBlock>,
        term_location: Location,
    ) {
        let tcx = self.tcx();
        match target {
            Some(_) => {
                let dest_ty = destination.ty(body, tcx).ty;
                let dest_ty = self.normalize(dest_ty, term_location);
                let category = match destination.as_local() {
                    Some(RETURN_PLACE) => {
                        if let BorrowCheckContext {
                            universal_regions:
                                UniversalRegions {
                                    defining_ty:
                                        DefiningTy::Const(def_id, _)
                                        | DefiningTy::InlineConst(def_id, _),
                                    ..
                                },
                            ..
                        } = self.borrowck_context
                        {
                            if tcx.is_static(*def_id) {
                                ConstraintCategory::UseAsStatic
                            } else {
                                ConstraintCategory::UseAsConst
                            }
                        } else {
                            ConstraintCategory::Return(ReturnConstraint::Normal)
                        }
                    }
                    Some(l) if !body.local_decls[l].is_user_variable() => {
                        ConstraintCategory::Boring
                    }
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
                if !output_ty.is_privately_uninhabited(self.tcx(), self.param_env) {
                    span_mirbug!(self, term, "call to converging function {:?} w/o dest", sig);
                }
            }
        }
    }

    fn check_call_inputs(
        &mut self,
        body: &Body<'tcx>,
        term: &Terminator<'tcx>,
        sig: &ty::FnSig<'tcx>,
        args: &[Operand<'tcx>],
        term_location: Location,
        from_hir_call: bool,
    ) {
        debug!("check_call_inputs({:?}, {:?})", sig, args);
        if args.len() < sig.inputs().len() || (args.len() > sig.inputs().len() && !sig.c_variadic) {
            span_mirbug!(self, term, "call to {:?} with wrong # of args", sig);
        }

        let func_ty = if let TerminatorKind::Call { func, .. } = &term.kind {
            Some(func.ty(body, self.infcx.tcx))
        } else {
            None
        };
        debug!(?func_ty);

        for (n, (fn_arg, op_arg)) in iter::zip(sig.inputs(), args).enumerate() {
            let op_arg_ty = op_arg.ty(body, self.tcx());

            let op_arg_ty = self.normalize(op_arg_ty, term_location);
            let category = if from_hir_call {
                ConstraintCategory::CallArgument(self.infcx.tcx.erase_regions(func_ty))
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

    fn check_iscleanup(&mut self, body: &Body<'tcx>, block_data: &BasicBlockData<'tcx>) {
        let is_cleanup = block_data.is_cleanup;
        self.last_span = block_data.terminator().source_info.span;
        match block_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                self.assert_iscleanup(body, block_data, target, is_cleanup)
            }
            TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets.all_targets() {
                    self.assert_iscleanup(body, block_data, *target, is_cleanup);
                }
            }
            TerminatorKind::Resume => {
                if !is_cleanup {
                    span_mirbug!(self, block_data, "resume on non-cleanup block!")
                }
            }
            TerminatorKind::Abort => {
                if !is_cleanup {
                    span_mirbug!(self, block_data, "abort on non-cleanup block!")
                }
            }
            TerminatorKind::Return => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "return on cleanup block")
                }
            }
            TerminatorKind::GeneratorDrop { .. } => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "generator_drop in cleanup block")
                }
            }
            TerminatorKind::Yield { resume, drop, .. } => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "yield in cleanup block")
                }
                self.assert_iscleanup(body, block_data, resume, is_cleanup);
                if let Some(drop) = drop {
                    self.assert_iscleanup(body, block_data, drop, is_cleanup);
                }
            }
            TerminatorKind::Unreachable => {}
            TerminatorKind::Drop { target, unwind, .. }
            | TerminatorKind::Assert { target, cleanup: unwind, .. } => {
                self.assert_iscleanup(body, block_data, target, is_cleanup);
                if let Some(unwind) = unwind {
                    if is_cleanup {
                        span_mirbug!(self, block_data, "unwind on cleanup block")
                    }
                    self.assert_iscleanup(body, block_data, unwind, true);
                }
            }
            TerminatorKind::Call { ref target, cleanup, .. } => {
                if let &Some(target) = target {
                    self.assert_iscleanup(body, block_data, target, is_cleanup);
                }
                if let Some(cleanup) = cleanup {
                    if is_cleanup {
                        span_mirbug!(self, block_data, "cleanup on cleanup block")
                    }
                    self.assert_iscleanup(body, block_data, cleanup, true);
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                self.assert_iscleanup(body, block_data, real_target, is_cleanup);
                self.assert_iscleanup(body, block_data, imaginary_target, is_cleanup);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                self.assert_iscleanup(body, block_data, real_target, is_cleanup);
                if let Some(unwind) = unwind {
                    if is_cleanup {
                        span_mirbug!(self, block_data, "cleanup in cleanup block via false unwind");
                    }
                    self.assert_iscleanup(body, block_data, unwind, true);
                }
            }
            TerminatorKind::InlineAsm { destination, cleanup, .. } => {
                if let Some(target) = destination {
                    self.assert_iscleanup(body, block_data, target, is_cleanup);
                }
                if let Some(cleanup) = cleanup {
                    if is_cleanup {
                        span_mirbug!(self, block_data, "cleanup on cleanup block")
                    }
                    self.assert_iscleanup(body, block_data, cleanup, true);
                }
            }
        }
    }

    fn assert_iscleanup(
        &mut self,
        body: &Body<'tcx>,
        ctxt: &dyn fmt::Debug,
        bb: BasicBlock,
        iscleanuppad: bool,
    ) {
        if body[bb].is_cleanup != iscleanuppad {
            span_mirbug!(self, ctxt, "cleanuppad mismatch: {:?} should be {:?}", bb, iscleanuppad);
        }
    }

    fn check_local(&mut self, body: &Body<'tcx>, local: Local, local_decl: &LocalDecl<'tcx>) {
        match body.local_kind(local) {
            LocalKind::ReturnPointer | LocalKind::Arg => {
                // return values of normal functions are required to be
                // sized by typeck, but return values of ADT constructors are
                // not because we don't include a `Self: Sized` bounds on them.
                //
                // Unbound parts of arguments were never required to be Sized
                // - maybe we should make that a warning.
                return;
            }
            LocalKind::Temp => {}
        }

        // When `unsized_fn_params` or `unsized_locals` is enabled, only function calls
        // and nullary ops are checked in `check_call_dest`.
        if !self.unsized_feature_enabled() {
            let span = local_decl.source_info.span;
            let ty = local_decl.ty;
            self.ensure_place_sized(ty, span);
        }
    }

    fn ensure_place_sized(&mut self, ty: Ty<'tcx>, span: Span) {
        let tcx = self.tcx();

        // Erase the regions from `ty` to get a global type. The
        // `Sized` bound in no way depends on precise regions, so this
        // shouldn't affect `is_sized`.
        let erased_ty = tcx.erase_regions(ty);
        if !erased_ty.is_sized(tcx, self.param_env) {
            // in current MIR construction, all non-control-flow rvalue
            // expressions evaluate through `as_temp` or `into` a return
            // slot or local, so to find all unsized rvalues it is enough
            // to check all temps, return slots and locals.
            if self.reported_errors.replace((ty, span)).is_none() {
                // While this is located in `nll::typeck` this error is not
                // an NLL error, it's a required check to prevent creation
                // of unsized rvalues in a call expression.
                self.tcx().sess.emit_err(MoveUnsized { ty, span });
            }
        }
    }

    fn aggregate_field_ty(
        &mut self,
        ak: &AggregateKind<'tcx>,
        field_index: usize,
        location: Location,
    ) -> Result<Ty<'tcx>, FieldAccessError> {
        let tcx = self.tcx();

        match *ak {
            AggregateKind::Adt(adt_did, variant_index, substs, _, active_field_index) => {
                let def = tcx.adt_def(adt_did);
                let variant = &def.variant(variant_index);
                let adj_field_index = active_field_index.unwrap_or(field_index);
                if let Some(field) = variant.fields.get(adj_field_index) {
                    Ok(self.normalize(field.ty(tcx, substs), location))
                } else {
                    Err(FieldAccessError::OutOfRange { field_count: variant.fields.len() })
                }
            }
            AggregateKind::Closure(_, substs) => {
                match substs.as_closure().upvar_tys().nth(field_index) {
                    Some(ty) => Ok(ty),
                    None => Err(FieldAccessError::OutOfRange {
                        field_count: substs.as_closure().upvar_tys().count(),
                    }),
                }
            }
            AggregateKind::Generator(_, substs, _) => {
                // It doesn't make sense to look at a field beyond the prefix;
                // these require a variant index, and are not initialized in
                // aggregate rvalues.
                match substs.as_generator().prefix_tys().nth(field_index) {
                    Some(ty) => Ok(ty),
                    None => Err(FieldAccessError::OutOfRange {
                        field_count: substs.as_generator().prefix_tys().count(),
                    }),
                }
            }
            AggregateKind::Array(ty) => Ok(ty),
            AggregateKind::Tuple => {
                unreachable!("This should have been covered in check_rvalues");
            }
        }
    }

    fn check_operand(&mut self, op: &Operand<'tcx>, location: Location) {
        debug!(?op, ?location, "check_operand");

        if let Operand::Constant(constant) = op {
            let maybe_uneval = match constant.literal {
                ConstantKind::Val(..) | ConstantKind::Ty(_) => None,
                ConstantKind::Unevaluated(uv, _) => Some(uv),
            };

            if let Some(uv) = maybe_uneval {
                if uv.promoted.is_none() {
                    let tcx = self.tcx();
                    let def_id = uv.def.def_id_for_type_of();
                    if tcx.def_kind(def_id) == DefKind::InlineConst {
                        let def_id = def_id.expect_local();
                        let predicates =
                            self.prove_closure_bounds(tcx, def_id, uv.substs, location);
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

    #[instrument(skip(self, body), level = "debug")]
    fn check_rvalue(&mut self, body: &Body<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx();
        let span = body.source_info(location).span;

        match rvalue {
            Rvalue::Aggregate(ak, ops) => {
                for op in ops {
                    self.check_operand(op, location);
                }
                self.check_aggregate_rvalue(&body, rvalue, ak, ops, location)
            }

            Rvalue::Repeat(operand, len) => {
                self.check_operand(operand, location);

                // If the length cannot be evaluated we must assume that the length can be larger
                // than 1.
                // If the length is larger than 1, the repeat expression will need to copy the
                // element, so we require the `Copy` trait.
                if len.try_eval_target_usize(tcx, self.param_env).map_or(true, |len| len > 1) {
                    match operand {
                        Operand::Copy(..) | Operand::Constant(..) => {
                            // These are always okay: direct use of a const, or a value that can evidently be copied.
                        }
                        Operand::Move(place) => {
                            // Make sure that repeated elements implement `Copy`.
                            let ty = place.ty(body, tcx).ty;
                            let trait_ref = tcx.at(span).mk_trait_ref(LangItem::Copy, [ty]);

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
                let trait_ref = tcx.at(span).mk_trait_ref(LangItem::Sized, [ty]);

                self.prove_trait_ref(
                    trait_ref,
                    location.to_locations(),
                    ConstraintCategory::SizedBound,
                );
            }

            Rvalue::ShallowInitBox(operand, ty) => {
                self.check_operand(operand, location);

                let trait_ref = tcx.at(span).mk_trait_ref(LangItem::Sized, [*ty]);

                self.prove_trait_ref(
                    trait_ref,
                    location.to_locations(),
                    ConstraintCategory::SizedBound,
                );
            }

            Rvalue::Cast(cast_kind, op, ty) => {
                self.check_operand(op, location);

                match cast_kind {
                    CastKind::Pointer(PointerCast::ReifyFnPointer) => {
                        let fn_sig = op.ty(body, tcx).fn_sig(tcx);

                        // The type that we see in the fcx is like
                        // `foo::<'a, 'b>`, where `foo` is the path to a
                        // function definition. When we extract the
                        // signature, it comes from the `fn_sig` query,
                        // and hence may contain unnormalized results.
                        let fn_sig = self.normalize(fn_sig, location);

                        let ty_fn_ptr_from = tcx.mk_fn_ptr(fn_sig);

                        if let Err(terr) = self.eq_types(
                            *ty,
                            ty_fn_ptr_from,
                            location.to_locations(),
                            ConstraintCategory::Cast,
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

                    CastKind::Pointer(PointerCast::ClosureFnPointer(unsafety)) => {
                        let sig = match op.ty(body, tcx).kind() {
                            ty::Closure(_, substs) => substs.as_closure().sig(),
                            _ => bug!(),
                        };
                        let ty_fn_ptr_from = tcx.mk_fn_ptr(tcx.signature_unclosure(sig, *unsafety));

                        if let Err(terr) = self.eq_types(
                            *ty,
                            ty_fn_ptr_from,
                            location.to_locations(),
                            ConstraintCategory::Cast,
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

                    CastKind::Pointer(PointerCast::UnsafeFnPointer) => {
                        let fn_sig = op.ty(body, tcx).fn_sig(tcx);

                        // The type that we see in the fcx is like
                        // `foo::<'a, 'b>`, where `foo` is the path to a
                        // function definition. When we extract the
                        // signature, it comes from the `fn_sig` query,
                        // and hence may contain unnormalized results.
                        let fn_sig = self.normalize(fn_sig, location);

                        let ty_fn_ptr_from = tcx.safe_to_unsafe_fn_ty(fn_sig);

                        if let Err(terr) = self.eq_types(
                            *ty,
                            ty_fn_ptr_from,
                            location.to_locations(),
                            ConstraintCategory::Cast,
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

                    CastKind::Pointer(PointerCast::Unsize) => {
                        let &ty = ty;
                        let trait_ref = tcx
                            .at(span)
                            .mk_trait_ref(LangItem::CoerceUnsized, [op.ty(body, tcx), ty]);

                        self.prove_trait_ref(
                            trait_ref,
                            location.to_locations(),
                            ConstraintCategory::Cast,
                        );
                    }

                    CastKind::DynStar => {
                        // get the constraints from the target type (`dyn* Clone`)
                        //
                        // apply them to prove that the source type `Foo` implements `Clone` etc
                        let (existential_predicates, region) = match ty.kind() {
                            Dynamic(predicates, region, ty::DynStar) => (predicates, region),
                            _ => panic!("Invalid dyn* cast_ty"),
                        };

                        let self_ty = op.ty(body, tcx);

                        self.prove_predicates(
                            existential_predicates
                                .iter()
                                .map(|predicate| predicate.with_self_ty(tcx, self_ty)),
                            location.to_locations(),
                            ConstraintCategory::Cast,
                        );

                        let outlives_predicate =
                            tcx.mk_predicate(Binder::dummy(ty::PredicateKind::Clause(
                                ty::Clause::TypeOutlives(ty::OutlivesPredicate(self_ty, *region)),
                            )));
                        self.prove_predicate(
                            outlives_predicate,
                            location.to_locations(),
                            ConstraintCategory::Cast,
                        );
                    }

                    CastKind::Pointer(PointerCast::MutToConstPointer) => {
                        let ty::RawPtr(ty::TypeAndMut {
                            ty: ty_from,
                            mutbl: hir::Mutability::Mut,
                        }) = op.ty(body, tcx).kind() else {
                            span_mirbug!(
                                self,
                                rvalue,
                                "unexpected base type for cast {:?}",
                                ty,
                            );
                            return;
                        };
                        let ty::RawPtr(ty::TypeAndMut {
                            ty: ty_to,
                            mutbl: hir::Mutability::Not,
                        }) = ty.kind() else {
                            span_mirbug!(
                                self,
                                rvalue,
                                "unexpected target type for cast {:?}",
                                ty,
                            );
                            return;
                        };
                        if let Err(terr) = self.sub_types(
                            *ty_from,
                            *ty_to,
                            location.to_locations(),
                            ConstraintCategory::Cast,
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

                    CastKind::Pointer(PointerCast::ArrayToPointer) => {
                        let ty_from = op.ty(body, tcx);

                        let opt_ty_elem_mut = match ty_from.kind() {
                            ty::RawPtr(ty::TypeAndMut { mutbl: array_mut, ty: array_ty }) => {
                                match array_ty.kind() {
                                    ty::Array(ty_elem, _) => Some((ty_elem, *array_mut)),
                                    _ => None,
                                }
                            }
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
                            ty::RawPtr(ty::TypeAndMut { mutbl: ty_to_mut, ty: ty_to }) => {
                                (ty_to, *ty_to_mut)
                            }
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

                        if let Err(terr) = self.sub_types(
                            *ty_elem,
                            *ty_to,
                            location.to_locations(),
                            ConstraintCategory::Cast,
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

                    CastKind::PointerExposeAddress => {
                        let ty_from = op.ty(body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Ptr(_) | CastTy::FnPtr), Some(CastTy::Int(_))) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid PointerExposeAddress cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }

                    CastKind::PointerFromExposedAddress => {
                        let ty_from = op.ty(body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Int(_)), Some(CastTy::Ptr(_))) => (),
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "Invalid PointerFromExposedAddress cast {:?} -> {:?}",
                                    ty_from,
                                    ty
                                )
                            }
                        }
                    }
                    CastKind::IntToInt => {
                        let ty_from = op.ty(body, tcx);
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
                        let ty_from = op.ty(body, tcx);
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
                        let ty_from = op.ty(body, tcx);
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
                        let ty_from = op.ty(body, tcx);
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
                        let ty_from = op.ty(body, tcx);
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
                        let ty_from = op.ty(body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(*ty);
                        match (cast_ty_from, cast_ty_to) {
                            (Some(CastTy::Ptr(_)), Some(CastTy::Ptr(_))) => (),
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
                self.add_reborrow_constraint(&body, location, *region, borrowed_place);
            }

            Rvalue::BinaryOp(
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge,
                box (left, right),
            ) => {
                self.check_operand(left, location);
                self.check_operand(right, location);

                let ty_left = left.ty(body, tcx);
                match ty_left.kind() {
                    // Types with regions are comparable if they have a common super-type.
                    ty::RawPtr(_) | ty::FnPtr(_) => {
                        let ty_right = right.ty(body, tcx);
                        let common_ty = self.infcx.next_ty_var(TypeVariableOrigin {
                            kind: TypeVariableOriginKind::MiscVariable,
                            span: body.source_info(location).span,
                        });
                        self.sub_types(
                            ty_left,
                            common_ty,
                            location.to_locations(),
                            ConstraintCategory::Boring,
                        )
                        .unwrap_or_else(|err| {
                            bug!("Could not equate type variable with {:?}: {:?}", ty_left, err)
                        });
                        if let Err(terr) = self.sub_types(
                            ty_right,
                            common_ty,
                            location.to_locations(),
                            ConstraintCategory::Boring,
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
                        if ty_left == right.ty(body, tcx) => {}
                    // Other types are compared by trait methods, not by
                    // `Rvalue::BinaryOp`.
                    _ => span_mirbug!(
                        self,
                        rvalue,
                        "unexpected comparison types {:?} and {:?}",
                        ty_left,
                        right.ty(body, tcx)
                    ),
                }
            }

            Rvalue::Use(operand) | Rvalue::UnaryOp(_, operand) => {
                self.check_operand(operand, location);
            }
            Rvalue::CopyForDeref(place) => {
                let op = &Operand::Copy(*place);
                self.check_operand(op, location);
            }

            Rvalue::BinaryOp(_, box (left, right))
            | Rvalue::CheckedBinaryOp(_, box (left, right)) => {
                self.check_operand(left, location);
                self.check_operand(right, location);
            }

            Rvalue::AddressOf(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Len(..)
            | Rvalue::Discriminant(..) => {}
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
            | Rvalue::AddressOf(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::CopyForDeref(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..) => None,

            Rvalue::Aggregate(aggregate, _) => match **aggregate {
                AggregateKind::Adt(_, _, _, user_ty, _) => user_ty,
                AggregateKind::Array(_) => None,
                AggregateKind::Tuple => None,
                AggregateKind::Closure(_, _) => None,
                AggregateKind::Generator(_, _, _) => None,
            },
        }
    }

    fn check_aggregate_rvalue(
        &mut self,
        body: &Body<'tcx>,
        rvalue: &Rvalue<'tcx>,
        aggregate_kind: &AggregateKind<'tcx>,
        operands: &[Operand<'tcx>],
        location: Location,
    ) {
        let tcx = self.tcx();

        self.prove_aggregate_predicates(aggregate_kind, location);

        if *aggregate_kind == AggregateKind::Tuple {
            // tuple rvalue field type is always the type of the op. Nothing to check here.
            return;
        }

        for (i, operand) in operands.iter().enumerate() {
            let field_ty = match self.aggregate_field_ty(aggregate_kind, i, location) {
                Ok(field_ty) => field_ty,
                Err(FieldAccessError::OutOfRange { field_count }) => {
                    span_mirbug!(
                        self,
                        rvalue,
                        "accessed field #{} but variant only has {}",
                        i,
                        field_count
                    );
                    continue;
                }
            };
            let operand_ty = operand.ty(body, tcx);
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
        body: &Body<'tcx>,
        location: Location,
        borrow_region: ty::Region<'tcx>,
        borrowed_place: &Place<'tcx>,
    ) {
        // These constraints are only meaningful during borrowck:
        let BorrowCheckContext { borrow_set, location_table, all_facts, constraints, .. } =
            self.borrowck_context;

        // In Polonius mode, we also push a `loan_issued_at` fact
        // linking the loan to the region (in some cases, though,
        // there is no loan associated with this borrow expression --
        // that occurs when we are borrowing an unsafe place, for
        // example).
        if let Some(all_facts) = all_facts {
            let _prof_timer = self.infcx.tcx.prof.generic_activity("polonius_fact_generation");
            if let Some(borrow_index) = borrow_set.get_index_of(&location) {
                let region_vid = borrow_region.to_region_vid();
                all_facts.loan_issued_at.push((
                    region_vid,
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

        let mut cursor = borrowed_place.projection.as_ref();
        let tcx = self.infcx.tcx;
        let field = path_utils::is_upvar_field_projection(
            tcx,
            &self.borrowck_context.upvars,
            borrowed_place.as_ref(),
            body,
        );
        let category = if let Some(field) = field {
            ConstraintCategory::ClosureUpvar(field)
        } else {
            ConstraintCategory::Boring
        };

        while let [proj_base @ .., elem] = cursor {
            cursor = proj_base;

            debug!("add_reborrow_constraint - iteration {:?}", elem);

            match elem {
                ProjectionElem::Deref => {
                    let base_ty = Place::ty_from(borrowed_place.local, proj_base, body, tcx).ty;

                    debug!("add_reborrow_constraint - base_ty = {:?}", base_ty);
                    match base_ty.kind() {
                        ty::Ref(ref_region, _, mutbl) => {
                            constraints.outlives_constraints.push(OutlivesConstraint {
                                sup: ref_region.to_region_vid(),
                                sub: borrow_region.to_region_vid(),
                                locations: location.to_locations(),
                                span: location.to_locations().span(body),
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
                | ProjectionElem::Subslice { .. } => {
                    // other field access
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
            AggregateKind::Adt(adt_did, _, substs, _, _) => {
                (adt_did, tcx.predicates_of(adt_did).instantiate(tcx, substs))
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
            AggregateKind::Closure(def_id, substs)
            | AggregateKind::Generator(def_id, substs, _) => {
                (def_id, self.prove_closure_bounds(tcx, def_id.expect_local(), substs, location))
            }

            AggregateKind::Array(_) | AggregateKind::Tuple => {
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
        substs: SubstsRef<'tcx>,
        location: Location,
    ) -> ty::InstantiatedPredicates<'tcx> {
        if let Some(closure_requirements) = &tcx.mir_borrowck(def_id).closure_requirements {
            constraint_conversion::ConstraintConversion::new(
                self.infcx,
                self.borrowck_context.universal_regions,
                self.region_bound_pairs,
                self.implicit_region_bound,
                self.param_env,
                location.to_locations(),
                DUMMY_SP,                   // irrelevant; will be overrided.
                ConstraintCategory::Boring, // same as above.
                &mut self.borrowck_context.constraints,
            )
            .apply_closure_requirements(
                &closure_requirements,
                def_id.to_def_id(),
                substs,
            );
        }

        // Now equate closure substs to regions inherited from `typeck_root_def_id`. Fixes #98589.
        let typeck_root_def_id = tcx.typeck_root_def_id(self.body.source.def_id());
        let typeck_root_substs = ty::InternalSubsts::identity_for_item(tcx, typeck_root_def_id);

        let parent_substs = match tcx.def_kind(def_id) {
            DefKind::Closure => substs.as_closure().parent_substs(),
            DefKind::Generator => substs.as_generator().parent_substs(),
            DefKind::InlineConst => substs.as_inline_const().parent_substs(),
            other => bug!("unexpected item {:?}", other),
        };
        let parent_substs = tcx.mk_substs(parent_substs);

        assert_eq!(typeck_root_substs.len(), parent_substs.len());
        if let Err(_) = self.eq_substs(
            typeck_root_substs,
            parent_substs,
            location.to_locations(),
            ConstraintCategory::BoringNoLocation,
        ) {
            span_mirbug!(
                self,
                def_id,
                "could not relate closure to parent {:?} != {:?}",
                typeck_root_substs,
                parent_substs
            );
        }

        tcx.predicates_of(def_id).instantiate(tcx, substs)
    }

    #[instrument(skip(self, body), level = "debug")]
    fn typeck_mir(&mut self, body: &Body<'tcx>) {
        self.last_span = body.span;
        debug!(?body.span);

        for (local, local_decl) in body.local_decls.iter_enumerated() {
            self.check_local(&body, local, local_decl);
        }

        for (block, block_data) in body.basic_blocks.iter_enumerated() {
            let mut location = Location { block, statement_index: 0 };
            for stmt in &block_data.statements {
                if !stmt.source_info.span.is_dummy() {
                    self.last_span = stmt.source_info.span;
                }
                self.check_stmt(body, stmt, location);
                location.statement_index += 1;
            }

            self.check_terminator(&body, block_data.terminator(), location);
            self.check_iscleanup(&body, block_data);
        }
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
    pub obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> TypeOp<'tcx> for InstantiateOpaqueType<'tcx> {
    type Output = ();
    /// We use this type itself to store the information used
    /// when reporting errors. Since this is not a query, we don't
    /// re-run anything during error reporting - we just use the information
    /// we saved to help extract an error from the already-existing region
    /// constraints in our `InferCtxt`
    type ErrorInfo = InstantiateOpaqueType<'tcx>;

    fn fully_perform(mut self, infcx: &InferCtxt<'tcx>) -> Fallible<TypeOpOutput<'tcx, Self>> {
        let (mut output, region_constraints) = scrape_region_constraints(infcx, || {
            Ok(InferOk { value: (), obligations: self.obligations.clone() })
        })?;
        self.region_constraints = Some(region_constraints);
        output.error_info = Some(self);
        Ok(output)
    }
}
