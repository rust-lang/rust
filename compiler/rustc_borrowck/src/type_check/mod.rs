//! This pass type-checks the MIR to ensure it is not broken.

use std::rc::Rc;
use std::{fmt, iter, mem};

use either::Either;

use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::vec_map::VecMap;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::lang_items::LangItem;
use rustc_index::vec::{Idx, IndexVec};
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::infer::opaque_types::OpaqueTypeDecl;
use rustc_infer::infer::outlives::env::RegionBoundPairs;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{
    InferCtxt, InferOk, LateBoundRegionConversionTime, NllRegionVariableOrigin,
};
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::AssertKind;
use rustc_middle::mir::*;
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::cast::CastTy;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef, UserSubsts};
use rustc_middle::ty::{
    self, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations, OpaqueTypeKey, RegionVid,
    ToPredicate, Ty, TyCtxt, UserType, UserTypeAnnotationIndex, WithConstness,
};
use rustc_span::def_id::CRATE_DEF_ID;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::VariantIdx;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::error_reporting::InferCtxtExt as _;
use rustc_trait_selection::traits::query::type_op;
use rustc_trait_selection::traits::query::type_op::custom::CustomTypeOp;
use rustc_trait_selection::traits::query::Fallible;
use rustc_trait_selection::traits::{self, ObligationCause, PredicateObligations};

use rustc_const_eval::transform::{
    check_consts::ConstCx, promote_consts::is_const_fn_in_array_repeat_expression,
};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::ResultsCursor;

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
    region_infer::{ClosureRegionRequirementsExt, TypeTest},
    type_check::free_region_relations::{CreateResult, UniversalRegionRelations},
    universal_regions::{DefiningTy, UniversalRegions},
    Upvar,
};

macro_rules! span_mirbug {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        $crate::type_check::mirbug(
            $context.tcx(),
            $context.last_span,
            &format!(
                "broken MIR in {:?} ({:?}): {}",
                $context.body.source.def_id(),
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
crate mod liveness;
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
    infcx: &InferCtxt<'_, 'tcx>,
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
) -> MirTypeckResults<'tcx> {
    let implicit_region_bound = infcx.tcx.mk_region(ty::ReVar(universal_regions.fr_fn_body));
    let mut universe_causes = FxHashMap::default();
    universe_causes.insert(ty::UniverseIndex::from_u32(0), UniverseInfo::other());
    let mut constraints = MirTypeckRegionConstraints {
        placeholder_indices: PlaceholderIndices::default(),
        placeholder_index_to_region: IndexVec::default(),
        liveness_constraints: LivenessValues::new(elements.clone()),
        outlives_constraints: OutlivesConstraintSet::default(),
        member_constraints: MemberConstraintSet::default(),
        closure_bounds_mapping: Default::default(),
        type_tests: Vec::default(),
        universe_causes,
    };

    let CreateResult {
        universal_region_relations,
        region_bound_pairs,
        normalized_inputs_and_output,
    } = free_region_relations::create(
        infcx,
        param_env,
        Some(implicit_region_bound),
        universal_regions,
        &mut constraints,
    );

    for u in ty::UniverseIndex::ROOT..infcx.universe() {
        let info = UniverseInfo::other();
        constraints.universe_causes.insert(u, info);
    }

    let mut borrowck_context = BorrowCheckContext {
        universal_regions,
        location_table,
        borrow_set,
        all_facts,
        constraints: &mut constraints,
        upvars,
    };

    let opaque_type_values = type_check_internal(
        infcx,
        param_env,
        body,
        promoted,
        &region_bound_pairs,
        implicit_region_bound,
        &mut borrowck_context,
        |mut cx| {
            cx.equate_inputs_and_outputs(&body, universal_regions, &normalized_inputs_and_output);
            liveness::generate(&mut cx, body, elements, flow_inits, move_data, location_table);

            translate_outlives_facts(&mut cx);
            let opaque_type_values = mem::take(&mut infcx.inner.borrow_mut().opaque_types);

            opaque_type_values
                .into_iter()
                .filter_map(|(opaque_type_key, mut decl)| {
                    decl.concrete_ty = infcx.resolve_vars_if_possible(decl.concrete_ty);
                    trace!(
                        "finalized opaque type {:?} to {:#?}",
                        opaque_type_key,
                        decl.concrete_ty.kind()
                    );
                    if decl.concrete_ty.has_infer_types_or_consts() {
                        infcx.tcx.sess.delay_span_bug(
                            body.span,
                            &format!("could not resolve {:#?}", decl.concrete_ty.kind()),
                        );
                        decl.concrete_ty = infcx.tcx.ty_error();
                    }
                    let concrete_is_opaque = if let ty::Opaque(def_id, _) = decl.concrete_ty.kind()
                    {
                        *def_id == opaque_type_key.def_id
                    } else {
                        false
                    };

                    if concrete_is_opaque {
                        // We're using an opaque `impl Trait` type without
                        // 'revealing' it. For example, code like this:
                        //
                        // type Foo = impl Debug;
                        // fn foo1() -> Foo { ... }
                        // fn foo2() -> Foo { foo1() }
                        //
                        // In `foo2`, we're not revealing the type of `Foo` - we're
                        // just treating it as the opaque type.
                        //
                        // When this occurs, we do *not* want to try to equate
                        // the concrete type with the underlying defining type
                        // of the opaque type - this will always fail, since
                        // the defining type of an opaque type is always
                        // some other type (e.g. not itself)
                        // Essentially, none of the normal obligations apply here -
                        // we're just passing around some unknown opaque type,
                        // without actually looking at the underlying type it
                        // gets 'revealed' into
                        debug!(
                            "eq_opaque_type_and_type: non-defining use of {:?}",
                            opaque_type_key.def_id,
                        );
                        None
                    } else {
                        Some((opaque_type_key, decl))
                    }
                })
                .collect()
        },
    );

    MirTypeckResults { constraints, universal_region_relations, opaque_type_values }
}

#[instrument(
    skip(infcx, body, promoted, region_bound_pairs, borrowck_context, extra),
    level = "debug"
)]
fn type_check_internal<'a, 'tcx, R>(
    infcx: &'a InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body: &'a Body<'tcx>,
    promoted: &'a IndexVec<Promoted, Body<'tcx>>,
    region_bound_pairs: &'a RegionBoundPairs<'tcx>,
    implicit_region_bound: ty::Region<'tcx>,
    borrowck_context: &'a mut BorrowCheckContext<'a, 'tcx>,
    extra: impl FnOnce(TypeChecker<'a, 'tcx>) -> R,
) -> R {
    let mut checker = TypeChecker::new(
        infcx,
        body,
        param_env,
        region_bound_pairs,
        implicit_region_bound,
        borrowck_context,
    );
    let errors_reported = {
        let mut verifier = TypeVerifier::new(&mut checker, body, promoted);
        verifier.visit_body(&body);
        verifier.errors_reported
    };

    if !errors_reported {
        // if verifier failed, don't do further checks to avoid ICEs
        checker.typeck_mir(body);
    }

    extra(checker)
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
    body: &'b Body<'tcx>,
    promoted: &'b IndexVec<Promoted, Body<'tcx>>,
    last_span: Span,
    errors_reported: bool,
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for TypeVerifier<'a, 'b, 'tcx> {
    fn visit_span(&mut self, span: &Span) {
        if !span.is_dummy() {
            self.last_span = *span;
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.sanitize_place(place, location, context);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
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

        if let Some(annotation_index) = constant.user_ty {
            if let Err(terr) = self.cx.relate_type_and_user_type(
                constant.literal.ty(),
                ty::Variance::Invariant,
                &UserTypeProjection { base: annotation_index, projs: vec![] },
                location.to_locations(),
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
                ConstantKind::Ty(ct) => match ct.val {
                    ty::ConstKind::Unevaluated(uv) => Some(uv),
                    _ => None,
                },
                _ => None,
            };
            if let Some(uv) = maybe_uneval {
                if let Some(promoted) = uv.promoted {
                    let check_err = |verifier: &mut TypeVerifier<'a, 'b, 'tcx>,
                                     promoted: &Body<'tcx>,
                                     ty,
                                     san_ty| {
                        if let Err(terr) = verifier.cx.eq_types(
                            ty,
                            san_ty,
                            location.to_locations(),
                            ConstraintCategory::Boring,
                        ) {
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
                    if let Err(terr) = self.cx.fully_perform_op(
                        location.to_locations(),
                        ConstraintCategory::Boring,
                        self.cx.param_env.and(type_op::ascribe_user_type::AscribeUserType::new(
                            constant.literal.ty(),
                            uv.def.did,
                            UserSubsts { substs: uv.substs(self.tcx()), user_self_ty: None },
                        )),
                    ) {
                        span_mirbug!(
                            self,
                            constant,
                            "bad constant type {:?} ({:?})",
                            constant,
                            terr
                        );
                    }
                }
            } else if let Some(static_def_id) = constant.check_static_ptr(tcx) {
                let unnormalized_ty = tcx.type_of(static_def_id);
                let locations = location.to_locations();
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
                let instantiated_predicates = tcx.predicates_of(def_id).instantiate(tcx, substs);
                self.cx.normalize_and_prove_instantiated_predicates(
                    def_id,
                    instantiated_predicates,
                    location.to_locations(),
                );
            }
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        let rval_ty = rvalue.ty(self.body, self.tcx());
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
                        rty
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
        body: &'b Body<'tcx>,
        promoted: &'b IndexVec<Promoted, Body<'tcx>>,
    ) -> Self {
        TypeVerifier { body, promoted, cx, last_span: body.span, errors_reported: false }
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

        let mut place_ty = PlaceTy::from_ty(self.body.local_decls[place.local].ty);

        for elem in place.projection.iter() {
            if place_ty.variant_index.is_none() {
                if place_ty.ty.references_error() {
                    assert!(self.errors_reported);
                    return PlaceTy::from_ty(self.tcx().ty_error());
                }
            }
            place_ty = self.sanitize_projection(place_ty, elem, place, location);
        }

        if let PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy) = context {
            let tcx = self.tcx();
            let trait_ref = ty::TraitRef {
                def_id: tcx.require_lang_item(LangItem::Copy, Some(self.last_span)),
                substs: tcx.mk_substs_trait(place_ty.ty, &[]),
            };

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

        let parent_body = mem::replace(&mut self.body, promoted_body);

        // Use new sets of constraints and closure bounds so that we can
        // modify their locations.
        let all_facts = &mut None;
        let mut constraints = Default::default();
        let mut closure_bounds = Default::default();
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
                &mut this.cx.borrowck_context.constraints.closure_bounds_mapping,
                &mut closure_bounds,
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

        self.body = parent_body;
        // Merge the outlives constraints back in, at the given location.
        swap_constraints(self);

        let locations = location.to_locations();
        for constraint in constraints.outlives().iter() {
            let mut constraint = constraint.clone();
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

        if !closure_bounds.is_empty() {
            let combined_bounds_mapping =
                closure_bounds.into_iter().flat_map(|(_, value)| value).collect();
            let existing = self
                .cx
                .borrowck_context
                .constraints
                .closure_bounds_mapping
                .insert(location, combined_bounds_mapping);
            assert!(existing.is_none(), "Multiple promoteds/closures at the same location.");
        }
    }

    fn sanitize_projection(
        &mut self,
        base: PlaceTy<'tcx>,
        pi: PlaceElem<'tcx>,
        place: &Place<'tcx>,
        location: Location,
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
                let index_ty = Place::from(i).ty(self.body, tcx).ty;
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
                        tcx.mk_array(inner, to - from)
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
                    if index.as_usize() >= adt_def.variants.len() {
                        PlaceTy::from_ty(span_mirbug_and_err!(
                            self,
                            place,
                            "cast to variant #{:?} but enum only has {:?}",
                            index,
                            adt_def.variants.len()
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
                match self.field_ty(place, base, field, location) {
                    Ok(ty) => {
                        let ty = self.cx.normalize(ty, location);
                        if let Err(terr) = self.cx.eq_types(
                            ty,
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
        }
    }

    fn error(&mut self) -> Ty<'tcx> {
        self.errors_reported = true;
        self.tcx().ty_error()
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
                ty::Adt(adt_def, substs) => (&adt_def.variants[variant_index], substs),
                ty::Generator(def_id, substs, _) => {
                    let mut variants = substs.as_generator().state_tys(def_id, tcx);
                    let mut variant = match variants.nth(variant_index.into()) {
                        Some(v) => v,
                        None => bug!(
                            "variant_index of generator out of range: {:?}/{:?}",
                            variant_index,
                            substs.as_generator().state_tys(def_id, tcx).count()
                        ),
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
                    (&adt_def.variants[VariantIdx::new(0)], substs)
                }
                ty::Closure(_, substs) => {
                    return match substs
                        .as_closure()
                        .tupled_upvars_ty()
                        .tuple_element_ty(field.index())
                    {
                        Some(ty) => Ok(ty),
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
                        Some(&ty) => Ok(ty.expect_ty()),
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
    infcx: &'a InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    last_span: Span,
    body: &'a Body<'tcx>,
    /// User type annotations are shared between the main MIR and the MIR of
    /// all of the promoted items.
    user_type_annotations: &'a CanonicalUserTypeAnnotations<'tcx>,
    region_bound_pairs: &'a RegionBoundPairs<'tcx>,
    implicit_region_bound: ty::Region<'tcx>,
    reported_errors: FxHashSet<(Ty<'tcx>, Span)>,
    borrowck_context: &'a mut BorrowCheckContext<'a, 'tcx>,
}

struct BorrowCheckContext<'a, 'tcx> {
    universal_regions: &'a UniversalRegions<'tcx>,
    location_table: &'a LocationTable,
    all_facts: &'a mut Option<AllFacts>,
    borrow_set: &'a BorrowSet<'tcx>,
    constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
    upvars: &'a [Upvar<'tcx>],
}

crate struct MirTypeckResults<'tcx> {
    crate constraints: MirTypeckRegionConstraints<'tcx>,
    crate universal_region_relations: Frozen<UniversalRegionRelations<'tcx>>,
    crate opaque_type_values: VecMap<OpaqueTypeKey<'tcx>, OpaqueTypeDecl<'tcx>>,
}

/// A collection of region constraints that must be satisfied for the
/// program to be considered well-typed.
crate struct MirTypeckRegionConstraints<'tcx> {
    /// Maps from a `ty::Placeholder` to the corresponding
    /// `PlaceholderIndex` bit that we will use for it.
    ///
    /// To keep everything in sync, do not insert this set
    /// directly. Instead, use the `placeholder_region` helper.
    crate placeholder_indices: PlaceholderIndices,

    /// Each time we add a placeholder to `placeholder_indices`, we
    /// also create a corresponding "representative" region vid for
    /// that wraps it. This vector tracks those. This way, when we
    /// convert the same `ty::RePlaceholder(p)` twice, we can map to
    /// the same underlying `RegionVid`.
    crate placeholder_index_to_region: IndexVec<PlaceholderIndex, ty::Region<'tcx>>,

    /// In general, the type-checker is not responsible for enforcing
    /// liveness constraints; this job falls to the region inferencer,
    /// which performs a liveness analysis. However, in some limited
    /// cases, the MIR type-checker creates temporary regions that do
    /// not otherwise appear in the MIR -- in particular, the
    /// late-bound regions that it instantiates at call-sites -- and
    /// hence it must report on their liveness constraints.
    crate liveness_constraints: LivenessValues<RegionVid>,

    crate outlives_constraints: OutlivesConstraintSet<'tcx>,

    crate member_constraints: MemberConstraintSet<'tcx, RegionVid>,

    crate closure_bounds_mapping:
        FxHashMap<Location, FxHashMap<(RegionVid, RegionVid), (ConstraintCategory, Span)>>,

    crate universe_causes: FxHashMap<ty::UniverseIndex, UniverseInfo<'tcx>>,

    crate type_tests: Vec<TypeTest<'tcx>>,
}

impl MirTypeckRegionConstraints<'tcx> {
    fn placeholder_region(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
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
    /// ```
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
        infcx: &'a InferCtxt<'a, 'tcx>,
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

    fn unsized_feature_enabled(&self) -> bool {
        let features = self.tcx().features();
        features.unsized_locals || features.unsized_fn_params
    }

    /// Equate the inferred type and the annotated type for user type annotations
    fn check_user_type_annotations(&mut self) {
        debug!(
            "check_user_type_annotations: user_type_annotations={:?}",
            self.user_type_annotations
        );
        for user_annotation in self.user_type_annotations {
            let CanonicalUserTypeAnnotation { span, ref user_ty, inferred_ty } = *user_annotation;
            let inferred_ty = self.normalize(inferred_ty, Locations::All(span));
            let annotation = self.instantiate_canonical_with_fresh_inference_vars(span, user_ty);
            match annotation {
                UserType::Ty(mut ty) => {
                    ty = self.normalize(ty, Locations::All(span));

                    if let Err(terr) = self.eq_types(
                        ty,
                        inferred_ty,
                        Locations::All(span),
                        ConstraintCategory::BoringNoLocation,
                    ) {
                        span_mirbug!(
                            self,
                            user_annotation,
                            "bad user type ({:?} = {:?}): {:?}",
                            ty,
                            inferred_ty,
                            terr
                        );
                    }

                    self.prove_predicate(
                        ty::Binder::dummy(ty::PredicateKind::WellFormed(inferred_ty.into()))
                            .to_predicate(self.tcx()),
                        Locations::All(span),
                        ConstraintCategory::TypeAnnotation,
                    );
                }
                UserType::TypeOf(def_id, user_substs) => {
                    if let Err(terr) = self.fully_perform_op(
                        Locations::All(span),
                        ConstraintCategory::BoringNoLocation,
                        self.param_env.and(type_op::ascribe_user_type::AscribeUserType::new(
                            inferred_ty,
                            def_id,
                            user_substs,
                        )),
                    ) {
                        span_mirbug!(
                            self,
                            user_annotation,
                            "bad user type AscribeUserType({:?}, {:?} {:?}, type_of={:?}): {:?}",
                            inferred_ty,
                            def_id,
                            user_substs,
                            self.tcx().type_of(def_id),
                            terr,
                        );
                    }
                }
            }
        }
    }

    #[instrument(skip(self, data), level = "debug")]
    fn push_region_constraints(
        &mut self,
        locations: Locations,
        category: ConstraintCategory,
        data: &QueryRegionConstraints<'tcx>,
    ) {
        debug!("constraints generated: {:#?}", data);

        constraint_conversion::ConstraintConversion::new(
            self.infcx,
            self.borrowck_context.universal_regions,
            self.region_bound_pairs,
            Some(self.implicit_region_bound),
            self.param_env,
            locations,
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
        category: ConstraintCategory,
    ) -> Fallible<()> {
        // Use this order of parameters because the sup type is usually the
        // "expected" type in diagnostics.
        self.relate_types(sup, ty::Variance::Contravariant, sub, locations, category)
    }

    fn eq_types(
        &mut self,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        locations: Locations,
        category: ConstraintCategory,
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
        category: ConstraintCategory,
    ) -> Fallible<()> {
        let annotated_type = self.user_type_annotations[user_ty.base].inferred_ty;
        let mut curr_projected_ty = PlaceTy::from_ty(annotated_type);

        let tcx = self.infcx.tcx;

        for proj in &user_ty.projs {
            let projected_ty = curr_projected_ty.projection_ty_core(
                tcx,
                self.param_env,
                proj,
                |this, field, &()| {
                    let ty = this.field_ty(tcx, field);
                    self.normalize(ty, locations)
                },
            );
            curr_projected_ty = projected_ty;
        }
        debug!(
            "user_ty base: {:?} freshened: {:?} projs: {:?} yields: {:?}",
            user_ty.base, annotated_type, user_ty.projs, curr_projected_ty
        );

        let ty = curr_projected_ty.ty;
        self.relate_types(ty, v.xform(ty::Variance::Contravariant), a, locations, category)?;

        Ok(())
    }

    /// Equates a type `anon_ty` that may contain opaque types whose
    /// values are to be inferred by the MIR.
    ///
    /// The type `revealed_ty` contains the same type as `anon_ty`, but with the
    /// hidden types for impl traits revealed.
    ///
    /// # Example
    ///
    /// Consider a piece of code like
    ///
    /// ```rust
    /// type Foo<U> = impl Debug;
    ///
    /// fn foo<T: Debug>(t: T) -> Box<Foo<T>> {
    ///      Box::new((t, 22_u32))
    /// }
    /// ```
    ///
    /// Here, the function signature would be something like
    /// `fn(T) -> Box<impl Debug>`. The MIR return slot would have
    /// the type with the opaque type revealed, so `Box<(T, u32)>`.
    ///
    /// In terms of our function parameters:
    ///
    /// * `anon_ty` would be `Box<Foo<T>>` where `Foo<T>` is an opaque type
    ///   scoped to this function (note that it is parameterized by the
    ///   generics of `foo`). Note that `anon_ty` is not just the opaque type,
    ///   but the entire return type (which may contain opaque types within it).
    /// * `revealed_ty` would be `Box<(T, u32)>`
    #[instrument(skip(self), level = "debug")]
    fn eq_opaque_type_and_type(
        &mut self,
        revealed_ty: Ty<'tcx>,
        anon_ty: Ty<'tcx>,
        locations: Locations,
        category: ConstraintCategory,
    ) -> Fallible<()> {
        // Fast path for the common case.
        if !anon_ty.has_opaque_types() {
            if let Err(terr) = self.eq_types(anon_ty, revealed_ty, locations, category) {
                span_mirbug!(
                    self,
                    locations,
                    "eq_opaque_type_and_type: `{:?}=={:?}` failed with `{:?}`",
                    revealed_ty,
                    anon_ty,
                    terr
                );
            }
            return Ok(());
        }

        let param_env = self.param_env;
        let body = self.body;
        let mir_def_id = body.source.def_id().expect_local();

        debug!(?mir_def_id);
        self.fully_perform_op(
            locations,
            category,
            CustomTypeOp::new(
                |infcx| {
                    let mut obligations = ObligationAccumulator::default();

                    let dummy_body_id = hir::CRATE_HIR_ID;

                    // Replace the opaque types defined by this function with
                    // inference variables, creating a map. In our example above,
                    // this would transform the type `Box<Foo<T>>` (where `Foo` is an opaque type)
                    // to `Box<?T>`, returning an `opaque_type_map` mapping `{Foo<T> -> ?T}`.
                    // (Note that the key of the map is both the def-id of `Foo` along with
                    // any generic parameters.)
                    let output_ty = obligations.add(infcx.instantiate_opaque_types(
                        dummy_body_id,
                        param_env,
                        anon_ty,
                        locations.span(body),
                    ));
                    debug!(?output_ty, ?revealed_ty);

                    // Make sure that the inferred types are well-formed. I'm
                    // not entirely sure this is needed (the HIR type check
                    // didn't do this) but it seems sensible to prevent opaque
                    // types hiding ill-formed types.
                    obligations.obligations.push(traits::Obligation::new(
                        ObligationCause::dummy(),
                        param_env,
                        ty::Binder::dummy(ty::PredicateKind::WellFormed(revealed_ty.into()))
                            .to_predicate(infcx.tcx),
                    ));
                    obligations.add(
                        infcx
                            .at(&ObligationCause::dummy(), param_env)
                            .eq(output_ty, revealed_ty)?,
                    );

                    debug!("equated");

                    Ok(InferOk { value: (), obligations: obligations.into_vec() })
                },
                || "input_output".to_string(),
            ),
        )?;

        // Finally, if we instantiated the anon types successfully, we
        // have to solve any bounds (e.g., `-> impl Iterator` needs to
        // prove that `T: Iterator` where `T` is the type we
        // instantiated it with).
        let opaque_type_map = self.infcx.inner.borrow().opaque_types.clone();
        for (opaque_type_key, opaque_decl) in opaque_type_map {
            self.fully_perform_op(
                locations,
                ConstraintCategory::OpaqueType,
                CustomTypeOp::new(
                    |infcx| {
                        infcx.constrain_opaque_type(opaque_type_key, &opaque_decl);
                        Ok(InferOk { value: (), obligations: vec![] })
                    },
                    || "opaque_type_map".to_string(),
                ),
            )?;
        }
        Ok(())
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[instrument(skip(self, body, location), level = "debug")]
    fn check_stmt(&mut self, body: &Body<'tcx>, stmt: &Statement<'tcx>, location: Location) {
        let tcx = self.tcx();
        match stmt.kind {
            StatementKind::Assign(box (ref place, ref rv)) => {
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
                        if matches!(
                            body.local_decls[l].local_info,
                            Some(box LocalInfo::AggregateTemp)
                        ) =>
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
                let place_ty = self.normalize(place_ty, location);
                let rv_ty = rv.ty(body, tcx);
                let rv_ty = self.normalize(rv_ty, location);
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
                    let trait_ref = ty::TraitRef {
                        def_id: tcx.require_lang_item(LangItem::Sized, Some(self.last_span)),
                        substs: tcx.mk_substs_trait(place_ty, &[]),
                    };
                    self.prove_trait_ref(
                        trait_ref,
                        location.to_locations(),
                        ConstraintCategory::SizedBound,
                    );
                }
            }
            StatementKind::SetDiscriminant { ref place, variant_index } => {
                let place_type = place.ty(body, tcx).ty;
                let adt = match place_type.kind() {
                    ty::Adt(adt, _) if adt.is_enum() => adt,
                    _ => {
                        span_bug!(
                            stmt.source_info.span,
                            "bad set discriminant ({:?} = {:?}): lhs is not an enum",
                            place,
                            variant_index
                        );
                    }
                };
                if variant_index.as_usize() >= adt.variants.len() {
                    span_bug!(
                        stmt.source_info.span,
                        "bad set discriminant ({:?} = {:?}): value of of range",
                        place,
                        variant_index
                    );
                };
            }
            StatementKind::AscribeUserType(box (ref place, ref projection), variance) => {
                let place_ty = place.ty(body, tcx).ty;
                if let Err(terr) = self.relate_type_and_user_type(
                    place_ty,
                    variance,
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
            StatementKind::CopyNonOverlapping(box rustc_middle::mir::CopyNonOverlapping {
                ..
            }) => span_bug!(
                stmt.source_info.span,
                "Unexpected StatementKind::CopyNonOverlapping, should only appear after lowering_intrinsics",
            ),
            StatementKind::FakeRead(..)
            | StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::LlvmInlineAsm { .. }
            | StatementKind::Retag { .. }
            | StatementKind::Coverage(..)
            | StatementKind::Nop => {}
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
        match term.kind {
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

            TerminatorKind::DropAndReplace { ref place, ref value, target: _, unwind: _ } => {
                let place_ty = place.ty(body, tcx).ty;
                let rv_ty = value.ty(body, tcx);

                let locations = term_location.to_locations();
                if let Err(terr) =
                    self.sub_types(rv_ty, place_ty, locations, ConstraintCategory::Assignment)
                {
                    span_mirbug!(
                        self,
                        term,
                        "bad DropAndReplace ({:?} = {:?}): {:?}",
                        place_ty,
                        rv_ty,
                        terr
                    );
                }
            }
            TerminatorKind::SwitchInt { ref discr, switch_ty, .. } => {
                self.check_operand(discr, term_location);

                let discr_ty = discr.ty(body, tcx);
                if let Err(terr) = self.sub_types(
                    discr_ty,
                    switch_ty,
                    term_location.to_locations(),
                    ConstraintCategory::Assignment,
                ) {
                    span_mirbug!(
                        self,
                        term,
                        "bad SwitchInt ({:?} on {:?}): {:?}",
                        switch_ty,
                        discr_ty,
                        terr
                    );
                }
                if !switch_ty.is_integral() && !switch_ty.is_char() && !switch_ty.is_bool() {
                    span_mirbug!(self, term, "bad SwitchInt discr ty {:?}", switch_ty);
                }
                // FIXME: check the values
            }
            TerminatorKind::Call { ref func, ref args, ref destination, from_hir_call, .. } => {
                self.check_operand(func, term_location);
                for arg in args {
                    self.check_operand(arg, term_location);
                }

                let func_ty = func.ty(body, tcx);
                debug!("check_terminator: call, func_ty={:?}", func_ty);
                let sig = match func_ty.kind() {
                    ty::FnDef(..) | ty::FnPtr(_) => func_ty.fn_sig(tcx),
                    _ => {
                        span_mirbug!(self, term, "call to non-function {:?}", func_ty);
                        return;
                    }
                };
                let (sig, map) = self.infcx.replace_bound_vars_with_fresh_vars(
                    term.source_info.span,
                    LateBoundRegionConversionTime::FnCall,
                    sig,
                );
                let sig = self.normalize(sig, term_location);
                self.check_call_dest(body, term, &sig, destination, term_location);

                self.prove_predicates(
                    sig.inputs_and_output
                        .iter()
                        .map(|ty| ty::Binder::dummy(ty::PredicateKind::WellFormed(ty.into()))),
                    term_location.to_locations(),
                    ConstraintCategory::Boring,
                );

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

                self.check_call_inputs(body, term, &sig, args, term_location, from_hir_call);
            }
            TerminatorKind::Assert { ref cond, ref msg, .. } => {
                self.check_operand(cond, term_location);

                let cond_ty = cond.ty(body, tcx);
                if cond_ty != tcx.types.bool {
                    span_mirbug!(self, term, "bad Assert ({:?}, not bool", cond_ty);
                }

                if let AssertKind::BoundsCheck { ref len, ref index } = *msg {
                    if len.ty(body, tcx) != tcx.types.usize {
                        span_mirbug!(self, len, "bounds-check length non-usize {:?}", len)
                    }
                    if index.ty(body, tcx) != tcx.types.usize {
                        span_mirbug!(self, index, "bounds-check index non-usize {:?}", index)
                    }
                }
            }
            TerminatorKind::Yield { ref value, .. } => {
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
        destination: &Option<(Place<'tcx>, BasicBlock)>,
        term_location: Location,
    ) {
        let tcx = self.tcx();
        match *destination {
            Some((ref dest, _target_block)) => {
                let dest_ty = dest.ty(body, tcx).ty;
                let dest_ty = self.normalize(dest_ty, term_location);
                let category = match dest.as_local() {
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
                if !self
                    .tcx()
                    .conservative_is_privately_uninhabited(self.param_env.and(sig.output()))
                {
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
        for (n, (fn_arg, op_arg)) in iter::zip(sig.inputs(), args).enumerate() {
            let op_arg_ty = op_arg.ty(body, self.tcx());
            let op_arg_ty = self.normalize(op_arg_ty, term_location);
            let category = if from_hir_call {
                ConstraintCategory::CallArgument
            } else {
                ConstraintCategory::Boring
            };
            if let Err(terr) =
                self.sub_types(op_arg_ty, fn_arg, term_location.to_locations(), category)
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
            | TerminatorKind::DropAndReplace { target, unwind, .. }
            | TerminatorKind::Assert { target, cleanup: unwind, .. } => {
                self.assert_iscleanup(body, block_data, target, is_cleanup);
                if let Some(unwind) = unwind {
                    if is_cleanup {
                        span_mirbug!(self, block_data, "unwind on cleanup block")
                    }
                    self.assert_iscleanup(body, block_data, unwind, true);
                }
            }
            TerminatorKind::Call { ref destination, cleanup, .. } => {
                if let &Some((_, target)) = destination {
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
            TerminatorKind::InlineAsm { destination, .. } => {
                if let Some(target) = destination {
                    self.assert_iscleanup(body, block_data, target, is_cleanup);
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
            LocalKind::Var | LocalKind::Temp => {}
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

        // Erase the regions from `ty` to get a global type.  The
        // `Sized` bound in no way depends on precise regions, so this
        // shouldn't affect `is_sized`.
        let erased_ty = tcx.erase_regions(ty);
        if !erased_ty.is_sized(tcx.at(span), self.param_env) {
            // in current MIR construction, all non-control-flow rvalue
            // expressions evaluate through `as_temp` or `into` a return
            // slot or local, so to find all unsized rvalues it is enough
            // to check all temps, return slots and locals.
            if self.reported_errors.replace((ty, span)).is_none() {
                let mut diag = struct_span_err!(
                    self.tcx().sess,
                    span,
                    E0161,
                    "cannot move a value of type {0}: the size of {0} \
                     cannot be statically determined",
                    ty
                );

                // While this is located in `nll::typeck` this error is not
                // an NLL error, it's a required check to prevent creation
                // of unsized rvalues in a call expression.
                diag.emit();
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
            AggregateKind::Adt(def, variant_index, substs, _, active_field_index) => {
                let variant = &def.variants[variant_index];
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
        if let Operand::Constant(constant) = op {
            let maybe_uneval = match constant.literal {
                ConstantKind::Ty(ct) => match ct.val {
                    ty::ConstKind::Unevaluated(uv) => Some(uv),
                    _ => None,
                },
                _ => None,
            };
            if let Some(uv) = maybe_uneval {
                if uv.promoted.is_none() {
                    let tcx = self.tcx();
                    let def_id = uv.def.def_id_for_type_of();
                    if tcx.def_kind(def_id) == DefKind::InlineConst {
                        let predicates = self.prove_closure_bounds(
                            tcx,
                            def_id.expect_local(),
                            uv.substs(tcx),
                            location,
                        );
                        self.normalize_and_prove_instantiated_predicates(
                            def_id,
                            predicates,
                            location.to_locations(),
                        );
                    }
                }
            }
        }
    }

    fn check_rvalue(&mut self, body: &Body<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx();

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
                if len.try_eval_usize(tcx, self.param_env).map_or(true, |len| len > 1) {
                    match operand {
                        Operand::Copy(..) | Operand::Constant(..) => {
                            // These are always okay: direct use of a const, or a value that can evidently be copied.
                        }
                        Operand::Move(place) => {
                            // Make sure that repeated elements implement `Copy`.
                            let span = body.source_info(location).span;
                            let ty = operand.ty(body, tcx);
                            if !self.infcx.type_is_copy_modulo_regions(self.param_env, ty, span) {
                                let ccx = ConstCx::new_with_param_env(tcx, body, self.param_env);
                                let is_const_fn =
                                    is_const_fn_in_array_repeat_expression(&ccx, &place, &body);

                                debug!("check_rvalue: is_const_fn={:?}", is_const_fn);

                                let def_id = body.source.def_id().expect_local();
                                let obligation = traits::Obligation::new(
                                    ObligationCause::new(
                                        span,
                                        self.tcx().hir().local_def_id_to_hir_id(def_id),
                                        traits::ObligationCauseCode::RepeatVec(is_const_fn),
                                    ),
                                    self.param_env,
                                    ty::Binder::dummy(ty::TraitRef::new(
                                        self.tcx().require_lang_item(
                                            LangItem::Copy,
                                            Some(self.last_span),
                                        ),
                                        tcx.mk_substs_trait(ty, &[]),
                                    ))
                                    .without_const()
                                    .to_predicate(self.tcx()),
                                );
                                self.infcx.report_selection_error(
                                    obligation.clone(),
                                    &obligation,
                                    &traits::SelectionError::Unimplemented,
                                    false,
                                );
                            }
                        }
                    }
                }
            }

            Rvalue::NullaryOp(_, ty) => {
                let trait_ref = ty::TraitRef {
                    def_id: tcx.require_lang_item(LangItem::Sized, Some(self.last_span)),
                    substs: tcx.mk_substs_trait(ty, &[]),
                };

                self.prove_trait_ref(
                    trait_ref,
                    location.to_locations(),
                    ConstraintCategory::SizedBound,
                );
            }

            Rvalue::ShallowInitBox(operand, ty) => {
                self.check_operand(operand, location);

                let trait_ref = ty::TraitRef {
                    def_id: tcx.require_lang_item(LangItem::Sized, Some(self.last_span)),
                    substs: tcx.mk_substs_trait(ty, &[]),
                };

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
                            ty,
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
                            ty,
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
                            ty,
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
                        let trait_ref = ty::TraitRef {
                            def_id: tcx
                                .require_lang_item(LangItem::CoerceUnsized, Some(self.last_span)),
                            substs: tcx.mk_substs_trait(op.ty(body, tcx), &[ty.into()]),
                        };

                        self.prove_trait_ref(
                            trait_ref,
                            location.to_locations(),
                            ConstraintCategory::Cast,
                        );
                    }

                    CastKind::Pointer(PointerCast::MutToConstPointer) => {
                        let ty_from = match op.ty(body, tcx).kind() {
                            ty::RawPtr(ty::TypeAndMut {
                                ty: ty_from,
                                mutbl: hir::Mutability::Mut,
                            }) => ty_from,
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "unexpected base type for cast {:?}",
                                    ty,
                                );
                                return;
                            }
                        };
                        let ty_to = match ty.kind() {
                            ty::RawPtr(ty::TypeAndMut {
                                ty: ty_to,
                                mutbl: hir::Mutability::Not,
                            }) => ty_to,
                            _ => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "unexpected target type for cast {:?}",
                                    ty,
                                );
                                return;
                            }
                        };
                        if let Err(terr) = self.sub_types(
                            ty_from,
                            ty_to,
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

                        let (ty_elem, ty_mut) = match opt_ty_elem_mut {
                            Some(ty_elem_mut) => ty_elem_mut,
                            None => {
                                span_mirbug!(
                                    self,
                                    rvalue,
                                    "ArrayToPointer cast from unexpected type {:?}",
                                    ty_from,
                                );
                                return;
                            }
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

                        if ty_to_mut == Mutability::Mut && ty_mut == Mutability::Not {
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
                            ty_elem,
                            ty_to,
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

                    CastKind::Misc => {
                        let ty_from = op.ty(body, tcx);
                        let cast_ty_from = CastTy::from_ty(ty_from);
                        let cast_ty_to = CastTy::from_ty(ty);
                        match (cast_ty_from, cast_ty_to) {
                            (None, _)
                            | (_, None | Some(CastTy::FnPtr))
                            | (Some(CastTy::Float), Some(CastTy::Ptr(_)))
                            | (Some(CastTy::Ptr(_) | CastTy::FnPtr), Some(CastTy::Float)) => {
                                span_mirbug!(self, rvalue, "Invalid cast {:?} -> {:?}", ty_from, ty,)
                            }
                            (
                                Some(CastTy::Int(_)),
                                Some(CastTy::Int(_) | CastTy::Float | CastTy::Ptr(_)),
                            )
                            | (Some(CastTy::Float), Some(CastTy::Int(_) | CastTy::Float))
                            | (Some(CastTy::Ptr(_)), Some(CastTy::Int(_) | CastTy::Ptr(_)))
                            | (Some(CastTy::FnPtr), Some(CastTy::Int(_) | CastTy::Ptr(_))) => (),
                        }
                    }
                }
            }

            Rvalue::Ref(region, _borrow_kind, borrowed_place) => {
                self.add_reborrow_constraint(&body, location, region, borrowed_place);
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
            let var_hir_id = self.borrowck_context.upvars[field.index()].place.get_root_variable();
            // FIXME(project-rfc-2229#8): Use Place for better diagnostics
            ConstraintCategory::ClosureUpvar(var_hir_id)
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
                                category,
                                variance_info: ty::VarianceDiagInfo::default(),
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

        let (def_id, instantiated_predicates) = match aggregate_kind {
            AggregateKind::Adt(def, _, substs, _, _) => {
                (def.did, tcx.predicates_of(def.did).instantiate(tcx, substs))
            }

            // For closures, we have some **extra requirements** we
            //
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
            // Despite the opacity of the previous parapgrah, this is
            // actually relatively easy to understand in terms of the
            // desugaring. A closure gets desugared to a struct, and
            // these extra requirements are basically like where
            // clauses on the struct.
            AggregateKind::Closure(def_id, substs)
            | AggregateKind::Generator(def_id, substs, _) => {
                (*def_id, self.prove_closure_bounds(tcx, def_id.expect_local(), substs, location))
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
        if let Some(ref closure_region_requirements) = tcx.mir_borrowck(def_id).closure_requirements
        {
            let closure_constraints = QueryRegionConstraints {
                outlives: closure_region_requirements.apply_requirements(
                    tcx,
                    def_id.to_def_id(),
                    substs,
                ),

                // Presently, closures never propagate member
                // constraints to their parents -- they are enforced
                // locally.  This is largely a non-issue as member
                // constraints only come from `-> impl Trait` and
                // friends which don't appear (thus far...) in
                // closures.
                member_constraints: vec![],
            };

            let bounds_mapping = closure_constraints
                .outlives
                .iter()
                .enumerate()
                .filter_map(|(idx, constraint)| {
                    let ty::OutlivesPredicate(k1, r2) =
                        constraint.no_bound_vars().unwrap_or_else(|| {
                            bug!("query_constraint {:?} contained bound vars", constraint,);
                        });

                    match k1.unpack() {
                        GenericArgKind::Lifetime(r1) => {
                            // constraint is r1: r2
                            let r1_vid = self.borrowck_context.universal_regions.to_region_vid(r1);
                            let r2_vid = self.borrowck_context.universal_regions.to_region_vid(r2);
                            let outlives_requirements =
                                &closure_region_requirements.outlives_requirements[idx];
                            Some((
                                (r1_vid, r2_vid),
                                (outlives_requirements.category, outlives_requirements.blame_span),
                            ))
                        }
                        GenericArgKind::Type(_) | GenericArgKind::Const(_) => None,
                    }
                })
                .collect();

            let existing = self
                .borrowck_context
                .constraints
                .closure_bounds_mapping
                .insert(location, bounds_mapping);
            assert!(existing.is_none(), "Multiple closures at the same location.");

            self.push_region_constraints(
                location.to_locations(),
                ConstraintCategory::ClosureBounds,
                &closure_constraints,
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

        for (block, block_data) in body.basic_blocks().iter_enumerated() {
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

#[derive(Debug, Default)]
struct ObligationAccumulator<'tcx> {
    obligations: PredicateObligations<'tcx>,
}

impl<'tcx> ObligationAccumulator<'tcx> {
    fn add<T>(&mut self, value: InferOk<'tcx, T>) -> T {
        let InferOk { value, obligations } = value;
        self.obligations.extend(obligations);
        value
    }

    fn into_vec(self) -> PredicateObligations<'tcx> {
        self.obligations
    }
}
