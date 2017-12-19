// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass type-checks the MIR to ensure it is not broken.
#![allow(unreachable_code)]

use borrow_check::nll::region_infer::Cause;
use borrow_check::nll::region_infer::ClosureRegionRequirementsExt;
use borrow_check::nll::universal_regions::UniversalRegions;
use dataflow::FlowAtLocation;
use dataflow::MaybeInitializedLvals;
use dataflow::move_paths::MoveData;
use rustc::hir::def_id::DefId;
use rustc::infer::{InferCtxt, InferOk, InferResult, LateBoundRegionConversionTime, UnitResult};
use rustc::infer::region_constraints::{GenericKind, RegionConstraintData};
use rustc::traits::{self, FulfillmentContext};
use rustc::ty::error::TypeError;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{self, ToPolyTraitRef, Ty, TyCtxt, TypeVariants};
use rustc::middle::const_val::ConstVal;
use rustc::mir::*;
use rustc::mir::tcx::PlaceTy;
use rustc::mir::visit::{PlaceContext, Visitor};
use std::fmt;
use syntax::ast;
use syntax_pos::{Span, DUMMY_SP};
use transform::{MirPass, MirSource};
use util::liveness::LivenessResults;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::Idx;

macro_rules! span_mirbug {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        $crate::borrow_check::nll::type_check::mirbug(
            $context.tcx(),
            $context.last_span,
            &format!(
                "broken MIR in {:?} ({:?}): {}",
                $context.body_id,
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

mod liveness;
mod input_output;

/// Type checks the given `mir` in the context of the inference
/// context `infcx`. Returns any region constraints that have yet to
/// be proven. This result is includes liveness constraints that
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
/// - `mir` -- MIR to type-check
/// - `mir_def_id` -- DefId from which the MIR is derived (must be local)
/// - `region_bound_pairs` -- the implied outlives obligations between type parameters
///   and lifetimes (e.g., `&'a T` implies `T: 'a`)
/// - `implicit_region_bound` -- a region which all generic parameters are assumed
///   to outlive; should represent the fn body
/// - `input_tys` -- fully liberated, but **not** normalized, expected types of the arguments;
///   the types of the input parameters found in the MIR itself will be equated with these
/// - `output_ty` -- fully liberaetd, but **not** normalized, expected return type;
///   the type for the RETURN_PLACE will be equated with this
/// - `liveness` -- results of a liveness computation on the MIR; used to create liveness
///   constraints for the regions in the types of variables
/// - `flow_inits` -- results of a maybe-init dataflow analysis
/// - `move_data` -- move-data constructed when performing the maybe-init dataflow analysis
pub(crate) fn type_check<'gcx, 'tcx>(
    infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'gcx>,
    mir: &Mir<'tcx>,
    mir_def_id: DefId,
    universal_regions: &UniversalRegions<'tcx>,
    liveness: &LivenessResults,
    flow_inits: &mut FlowAtLocation<MaybeInitializedLvals<'_, 'gcx, 'tcx>>,
    move_data: &MoveData<'tcx>,
) -> MirTypeckRegionConstraints<'tcx> {
    let body_id = infcx.tcx.hir.as_local_node_id(mir_def_id).unwrap();
    let implicit_region_bound = infcx.tcx.mk_region(ty::ReVar(universal_regions.fr_fn_body));
    type_check_internal(
        infcx,
        body_id,
        param_env,
        mir,
        &universal_regions.region_bound_pairs,
        Some(implicit_region_bound),
        &mut |cx| {
            liveness::generate(cx, mir, liveness, flow_inits, move_data);

            cx.equate_inputs_and_outputs(mir, mir_def_id, universal_regions);
        },
    )
}

fn type_check_internal<'gcx, 'tcx>(
    infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    body_id: ast::NodeId,
    param_env: ty::ParamEnv<'gcx>,
    mir: &Mir<'tcx>,
    region_bound_pairs: &[(ty::Region<'tcx>, GenericKind<'tcx>)],
    implicit_region_bound: Option<ty::Region<'tcx>>,
    extra: &mut FnMut(&mut TypeChecker<'_, 'gcx, 'tcx>),
) -> MirTypeckRegionConstraints<'tcx> {
    let mut checker = TypeChecker::new(
        infcx,
        body_id,
        param_env,
        region_bound_pairs,
        implicit_region_bound,
    );
    let errors_reported = {
        let mut verifier = TypeVerifier::new(&mut checker, mir);
        verifier.visit_mir(mir);
        verifier.errors_reported
    };

    if !errors_reported {
        // if verifier failed, don't do further checks to avoid ICEs
        checker.typeck_mir(mir);
    }

    extra(&mut checker);

    checker.constraints
}

fn mirbug(tcx: TyCtxt, span: Span, msg: &str) {
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
struct TypeVerifier<'a, 'b: 'a, 'gcx: 'b + 'tcx, 'tcx: 'b> {
    cx: &'a mut TypeChecker<'b, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    last_span: Span,
    body_id: ast::NodeId,
    errors_reported: bool,
}

impl<'a, 'b, 'gcx, 'tcx> Visitor<'tcx> for TypeVerifier<'a, 'b, 'gcx, 'tcx> {
    fn visit_span(&mut self, span: &Span) {
        if *span != DUMMY_SP {
            self.last_span = *span;
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.sanitize_place(place, location, context);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        self.super_constant(constant, location);
        self.sanitize_constant(constant, location);
        self.sanitize_type(constant, constant.ty);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        let rval_ty = rvalue.ty(self.mir, self.tcx());
        self.sanitize_type(rvalue, rval_ty);
    }

    fn visit_local_decl(&mut self, local: Local, local_decl: &LocalDecl<'tcx>) {
        self.super_local_decl(local, local_decl);
        self.sanitize_type(local_decl, local_decl.ty);
    }

    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        self.sanitize_type(&"return type", mir.return_ty());
        for local_decl in &mir.local_decls {
            self.sanitize_type(local_decl, local_decl.ty);
        }
        if self.errors_reported {
            return;
        }
        self.super_mir(mir);
    }
}

impl<'a, 'b, 'gcx, 'tcx> TypeVerifier<'a, 'b, 'gcx, 'tcx> {
    fn new(cx: &'a mut TypeChecker<'b, 'gcx, 'tcx>, mir: &'a Mir<'tcx>) -> Self {
        TypeVerifier {
            mir,
            body_id: cx.body_id,
            cx,
            last_span: mir.span,
            errors_reported: false,
        }
    }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.cx.infcx.tcx
    }

    fn sanitize_type(&mut self, parent: &fmt::Debug, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.has_escaping_regions() || ty.references_error() {
            span_mirbug_and_err!(self, parent, "bad type {:?}", ty)
        } else {
            ty
        }
    }

    /// Checks that the constant's `ty` field matches up with what
    /// would be expected from its literal.
    fn sanitize_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        debug!(
            "sanitize_constant(constant={:?}, location={:?})",
            constant,
            location
        );

        let expected_ty = match constant.literal {
            Literal::Value { value } => {
                // FIXME(#46702) -- We need some way to get the predicates
                // associated with the "pre-evaluated" form of the
                // constant. For example, consider that the constant
                // may have associated constant projections (`<Foo as
                // Trait<'a, 'b>>::SOME_CONST`) that impose
                // constraints on `'a` and `'b`. These constraints
                // would be lost if we just look at the normalized
                // value.
                if let ConstVal::Function(def_id, ..) = value.val {
                    let tcx = self.tcx();
                    let type_checker = &mut self.cx;

                    // FIXME -- For now, use the substitutions from
                    // `value.ty` rather than `value.val`. The
                    // renumberer will rewrite them to independent
                    // sets of regions; in principle, we ought to
                    // derive the type of the `value.val` from "first
                    // principles" and equate with value.ty, but as we
                    // are transitioning to the miri-based system, we
                    // don't have a handy function for that, so for
                    // now we just ignore `value.val` regions.
                    let substs = match value.ty.sty {
                        ty::TyFnDef(ty_def_id, substs) => {
                            assert_eq!(def_id, ty_def_id);
                            substs
                        }
                        _ => span_bug!(
                            self.last_span,
                            "unexpected type for constant function: {:?}",
                            value.ty
                        ),
                    };

                    let instantiated_predicates =
                        tcx.predicates_of(def_id).instantiate(tcx, substs);
                    let predicates =
                        type_checker.normalize(&instantiated_predicates.predicates, location);
                    type_checker.prove_predicates(&predicates, location);
                }

                value.ty
            }

            Literal::Promoted { .. } => {
                // FIXME -- promoted MIR return types reference
                // various "free regions" (e.g., scopes and things)
                // that they ought not to do. We have to figure out
                // how best to handle that -- probably we want treat
                // promoted MIR much like closures, renumbering all
                // their free regions and propagating constraints
                // upwards. We have the same acyclic guarantees, so
                // that should be possible. But for now, ignore them.
                //
                // let promoted_mir = &self.mir.promoted[index];
                // promoted_mir.return_ty()
                return;
            }
        };

        debug!("sanitize_constant: expected_ty={:?}", expected_ty);

        if let Err(terr) = self.cx
            .eq_types(expected_ty, constant.ty, location.at_self())
        {
            span_mirbug!(
                self,
                constant,
                "constant {:?} should have type {:?} but has {:?} ({:?})",
                constant,
                expected_ty,
                constant.ty,
                terr,
            );
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
        let place_ty = match *place {
            Place::Local(index) => PlaceTy::Ty {
                ty: self.mir.local_decls[index].ty,
            },
            Place::Static(box Static { def_id, ty: sty }) => {
                let sty = self.sanitize_type(place, sty);
                let ty = self.tcx().type_of(def_id);
                let ty = self.cx.normalize(&ty, location);
                if let Err(terr) = self.cx.eq_types(ty, sty, location.at_self()) {
                    span_mirbug!(
                        self,
                        place,
                        "bad static type ({:?}: {:?}): {:?}",
                        ty,
                        sty,
                        terr
                    );
                }
                PlaceTy::Ty { ty: sty }
            }
            Place::Projection(ref proj) => {
                let base_context = if context.is_mutating_use() {
                    PlaceContext::Projection(Mutability::Mut)
                } else {
                    PlaceContext::Projection(Mutability::Not)
                };
                let base_ty = self.sanitize_place(&proj.base, location, base_context);
                if let PlaceTy::Ty { ty } = base_ty {
                    if ty.references_error() {
                        assert!(self.errors_reported);
                        return PlaceTy::Ty {
                            ty: self.tcx().types.err,
                        };
                    }
                }
                self.sanitize_projection(base_ty, &proj.elem, place, location)
            }
        };
        if let PlaceContext::Copy = context {
            let ty = place_ty.to_ty(self.tcx());
            if self.cx
                .infcx
                .type_moves_by_default(self.cx.param_env, ty, DUMMY_SP)
            {
                span_mirbug!(self, place, "attempted copy of non-Copy type ({:?})", ty);
            }
        }
        place_ty
    }

    fn sanitize_projection(
        &mut self,
        base: PlaceTy<'tcx>,
        pi: &PlaceElem<'tcx>,
        place: &Place<'tcx>,
        location: Location,
    ) -> PlaceTy<'tcx> {
        debug!("sanitize_projection: {:?} {:?} {:?}", base, pi, place);
        let tcx = self.tcx();
        let base_ty = base.to_ty(tcx);
        match *pi {
            ProjectionElem::Deref => {
                let deref_ty = base_ty.builtin_deref(true, ty::LvaluePreference::NoPreference);
                PlaceTy::Ty {
                    ty: deref_ty.map(|t| t.ty).unwrap_or_else(|| {
                        span_mirbug_and_err!(self, place, "deref of non-pointer {:?}", base_ty)
                    }),
                }
            }
            ProjectionElem::Index(i) => {
                let index_ty = Place::Local(i).ty(self.mir, tcx).to_ty(tcx);
                if index_ty != tcx.types.usize {
                    PlaceTy::Ty {
                        ty: span_mirbug_and_err!(self, i, "index by non-usize {:?}", i),
                    }
                } else {
                    PlaceTy::Ty {
                        ty: base_ty.builtin_index().unwrap_or_else(|| {
                            span_mirbug_and_err!(self, place, "index of non-array {:?}", base_ty)
                        }),
                    }
                }
            }
            ProjectionElem::ConstantIndex { .. } => {
                // consider verifying in-bounds
                PlaceTy::Ty {
                    ty: base_ty.builtin_index().unwrap_or_else(|| {
                        span_mirbug_and_err!(self, place, "index of non-array {:?}", base_ty)
                    }),
                }
            }
            ProjectionElem::Subslice { from, to } => PlaceTy::Ty {
                ty: match base_ty.sty {
                    ty::TyArray(inner, size) => {
                        let size = size.val.to_const_int().unwrap().to_u64().unwrap();
                        let min_size = (from as u64) + (to as u64);
                        if let Some(rest_size) = size.checked_sub(min_size) {
                            tcx.mk_array(inner, rest_size)
                        } else {
                            span_mirbug_and_err!(
                                self,
                                place,
                                "taking too-small slice of {:?}",
                                base_ty
                            )
                        }
                    }
                    ty::TySlice(..) => base_ty,
                    _ => span_mirbug_and_err!(self, place, "slice of non-array {:?}", base_ty),
                },
            },
            ProjectionElem::Downcast(adt_def1, index) => match base_ty.sty {
                ty::TyAdt(adt_def, substs) if adt_def.is_enum() && adt_def == adt_def1 => {
                    if index >= adt_def.variants.len() {
                        PlaceTy::Ty {
                            ty: span_mirbug_and_err!(
                                self,
                                place,
                                "cast to variant #{:?} but enum only has {:?}",
                                index,
                                adt_def.variants.len()
                            ),
                        }
                    } else {
                        PlaceTy::Downcast {
                            adt_def,
                            substs,
                            variant_index: index,
                        }
                    }
                }
                _ => PlaceTy::Ty {
                    ty: span_mirbug_and_err!(
                        self,
                        place,
                        "can't downcast {:?} as {:?}",
                        base_ty,
                        adt_def1
                    ),
                },
            },
            ProjectionElem::Field(field, fty) => {
                let fty = self.sanitize_type(place, fty);
                match self.field_ty(place, base, field, location) {
                    Ok(ty) => if let Err(terr) = self.cx.eq_types(ty, fty, location.at_self()) {
                        span_mirbug!(
                            self,
                            place,
                            "bad field access ({:?}: {:?}): {:?}",
                            ty,
                            fty,
                            terr
                        );
                    },
                    Err(FieldAccessError::OutOfRange { field_count }) => span_mirbug!(
                        self,
                        place,
                        "accessed field #{} but variant only has {}",
                        field.index(),
                        field_count
                    ),
                }
                PlaceTy::Ty { ty: fty }
            }
        }
    }

    fn error(&mut self) -> Ty<'tcx> {
        self.errors_reported = true;
        self.tcx().types.err
    }

    fn field_ty(
        &mut self,
        parent: &fmt::Debug,
        base_ty: PlaceTy<'tcx>,
        field: Field,
        location: Location,
    ) -> Result<Ty<'tcx>, FieldAccessError> {
        let tcx = self.tcx();

        let (variant, substs) = match base_ty {
            PlaceTy::Downcast {
                adt_def,
                substs,
                variant_index,
            } => (&adt_def.variants[variant_index], substs),
            PlaceTy::Ty { ty } => match ty.sty {
                ty::TyAdt(adt_def, substs) if !adt_def.is_enum() => (&adt_def.variants[0], substs),
                ty::TyClosure(def_id, substs) => {
                    return match substs.upvar_tys(def_id, tcx).nth(field.index()) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.upvar_tys(def_id, tcx).count(),
                        }),
                    }
                }
                ty::TyGenerator(def_id, substs, _) => {
                    // Try upvars first. `field_tys` requires final optimized MIR.
                    if let Some(ty) = substs.upvar_tys(def_id, tcx).nth(field.index()) {
                        return Ok(ty);
                    }

                    return match substs.field_tys(def_id, tcx).nth(field.index()) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.field_tys(def_id, tcx).count() + 1,
                        }),
                    };
                }
                ty::TyTuple(tys, _) => {
                    return match tys.get(field.index()) {
                        Some(&ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: tys.len(),
                        }),
                    }
                }
                _ => {
                    return Ok(span_mirbug_and_err!(
                        self,
                        parent,
                        "can't project out of {:?}",
                        base_ty
                    ))
                }
            },
        };

        if let Some(field) = variant.fields.get(field.index()) {
            Ok(self.cx.normalize(&field.ty(tcx, substs), location))
        } else {
            Err(FieldAccessError::OutOfRange {
                field_count: variant.fields.len(),
            })
        }
    }
}

/// The MIR type checker. Visits the MIR and enforces all the
/// constraints needed for it to be valid and well-typed. Along the
/// way, it accrues region constraints -- these can later be used by
/// NLL region checking.
struct TypeChecker<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'gcx>,
    last_span: Span,
    body_id: ast::NodeId,
    region_bound_pairs: &'a [(ty::Region<'tcx>, GenericKind<'tcx>)],
    implicit_region_bound: Option<ty::Region<'tcx>>,
    reported_errors: FxHashSet<(Ty<'tcx>, Span)>,
    constraints: MirTypeckRegionConstraints<'tcx>,
}

/// A collection of region constraints that must be satisfied for the
/// program to be considered well-typed.
#[derive(Default)]
pub(crate) struct MirTypeckRegionConstraints<'tcx> {
    /// In general, the type-checker is not responsible for enforcing
    /// liveness constraints; this job falls to the region inferencer,
    /// which performs a liveness analysis. However, in some limited
    /// cases, the MIR type-checker creates temporary regions that do
    /// not otherwise appear in the MIR -- in particular, the
    /// late-bound regions that it instantiates at call-sites -- and
    /// hence it must report on their liveness constraints.
    pub liveness_set: Vec<(ty::Region<'tcx>, Location, Cause)>,

    /// During the course of type-checking, we will accumulate region
    /// constraints due to performing subtyping operations or solving
    /// traits. These are accumulated into this vector for later use.
    pub outlives_sets: Vec<OutlivesSet<'tcx>>,
}

/// Outlives relationships between regions and types created at a
/// particular point within the control-flow graph.
pub struct OutlivesSet<'tcx> {
    /// The locations associated with these constraints.
    pub locations: Locations,

    /// Constraints generated. In terms of the NLL RFC, when you have
    /// a constraint `R1: R2 @ P`, the data in there specifies things
    /// like `R1: R2`.
    pub data: RegionConstraintData<'tcx>,
}

#[derive(Copy, Clone, Debug)]
pub struct Locations {
    /// The location in the MIR that generated these constraints.
    /// This is intended for error reporting and diagnosis; the
    /// constraints may *take effect* at a distinct spot.
    pub from_location: Location,

    /// The constraints must be met at this location. In terms of the
    /// NLL RFC, when you have a constraint `R1: R2 @ P`, this field
    /// is the `P` value.
    pub at_location: Location,
}

impl<'a, 'gcx, 'tcx> TypeChecker<'a, 'gcx, 'tcx> {
    fn new(
        infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
        body_id: ast::NodeId,
        param_env: ty::ParamEnv<'gcx>,
        region_bound_pairs: &'a [(ty::Region<'tcx>, GenericKind<'tcx>)],
        implicit_region_bound: Option<ty::Region<'tcx>>,
    ) -> Self {
        TypeChecker {
            infcx,
            last_span: DUMMY_SP,
            body_id,
            param_env,
            region_bound_pairs,
            implicit_region_bound,
            reported_errors: FxHashSet(),
            constraints: MirTypeckRegionConstraints::default(),
        }
    }

    fn misc(&self, span: Span) -> traits::ObligationCause<'tcx> {
        traits::ObligationCause::misc(span, self.body_id)
    }

    fn fully_perform_op<OP, R>(
        &mut self,
        locations: Locations,
        op: OP,
    ) -> Result<R, TypeError<'tcx>>
    where
        OP: FnOnce(&mut Self) -> InferResult<'tcx, R>,
    {
        let mut fulfill_cx = FulfillmentContext::new();
        let InferOk { value, obligations } = self.infcx.commit_if_ok(|_| op(self))?;
        fulfill_cx.register_predicate_obligations(self.infcx, obligations);
        if let Err(e) = fulfill_cx.select_all_or_error(self.infcx) {
            span_mirbug!(self, "", "errors selecting obligation: {:?}", e);
        }

        self.infcx.process_registered_region_obligations(
            self.region_bound_pairs,
            self.implicit_region_bound,
            self.param_env,
            self.body_id,
        );

        let data = self.infcx.take_and_reset_region_constraints();
        if !data.is_empty() {
            self.constraints
                .outlives_sets
                .push(OutlivesSet { locations, data });
        }

        Ok(value)
    }

    fn sub_types(
        &mut self,
        sub: Ty<'tcx>,
        sup: Ty<'tcx>,
        locations: Locations,
    ) -> UnitResult<'tcx> {
        self.fully_perform_op(locations, |this| {
            this.infcx
                .at(&this.misc(this.last_span), this.param_env)
                .sup(sup, sub)
        })
    }

    fn eq_types(&mut self, a: Ty<'tcx>, b: Ty<'tcx>, locations: Locations) -> UnitResult<'tcx> {
        self.fully_perform_op(locations, |this| {
            this.infcx
                .at(&this.misc(this.last_span), this.param_env)
                .eq(b, a)
        })
    }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn check_stmt(&mut self, mir: &Mir<'tcx>, stmt: &Statement<'tcx>, location: Location) {
        debug!("check_stmt: {:?}", stmt);
        let tcx = self.tcx();
        match stmt.kind {
            StatementKind::Assign(ref place, ref rv) => {
                let place_ty = place.ty(mir, tcx).to_ty(tcx);
                let rv_ty = rv.ty(mir, tcx);
                if let Err(terr) =
                    self.sub_types(rv_ty, place_ty, location.at_successor_within_block())
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
                self.check_rvalue(mir, rv, location);
            }
            StatementKind::SetDiscriminant {
                ref place,
                variant_index,
            } => {
                let place_type = place.ty(mir, tcx).to_ty(tcx);
                let adt = match place_type.sty {
                    TypeVariants::TyAdt(adt, _) if adt.is_enum() => adt,
                    _ => {
                        span_bug!(
                            stmt.source_info.span,
                            "bad set discriminant ({:?} = {:?}): lhs is not an enum",
                            place,
                            variant_index
                        );
                    }
                };
                if variant_index >= adt.variants.len() {
                    span_bug!(
                        stmt.source_info.span,
                        "bad set discriminant ({:?} = {:?}): value of of range",
                        place,
                        variant_index
                    );
                };
            }
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::InlineAsm { .. }
            | StatementKind::EndRegion(_)
            | StatementKind::Validate(..)
            | StatementKind::Nop => {}
        }
    }

    fn check_terminator(
        &mut self,
        mir: &Mir<'tcx>,
        term: &Terminator<'tcx>,
        term_location: Location,
    ) {
        debug!("check_terminator: {:?}", term);
        let tcx = self.tcx();
        match term.kind {
            TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Unreachable
            | TerminatorKind::Drop { .. }
            | TerminatorKind::FalseEdges { .. } => {
                // no checks needed for these
            }

            TerminatorKind::DropAndReplace {
                ref location,
                ref value,
                target,
                unwind,
            } => {
                let place_ty = location.ty(mir, tcx).to_ty(tcx);
                let rv_ty = value.ty(mir, tcx);

                let locations = Locations {
                    from_location: term_location,
                    at_location: target.start_location(),
                };
                if let Err(terr) = self.sub_types(rv_ty, place_ty, locations) {
                    span_mirbug!(
                        self,
                        term,
                        "bad DropAndReplace ({:?} = {:?}): {:?}",
                        place_ty,
                        rv_ty,
                        terr
                    );
                }

                // Subtle: this assignment occurs at the start of
                // *both* blocks, so we need to ensure that it holds
                // at both locations.
                if let Some(unwind) = unwind {
                    let locations = Locations {
                        from_location: term_location,
                        at_location: unwind.start_location(),
                    };
                    if let Err(terr) = self.sub_types(rv_ty, place_ty, locations) {
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
            }
            TerminatorKind::SwitchInt {
                ref discr,
                switch_ty,
                ..
            } => {
                let discr_ty = discr.ty(mir, tcx);
                if let Err(terr) = self.sub_types(discr_ty, switch_ty, term_location.at_self()) {
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
            TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                ..
            } => {
                let func_ty = func.ty(mir, tcx);
                debug!("check_terminator: call, func_ty={:?}", func_ty);
                let sig = match func_ty.sty {
                    ty::TyFnDef(..) | ty::TyFnPtr(_) => func_ty.fn_sig(tcx),
                    _ => {
                        span_mirbug!(self, term, "call to non-function {:?}", func_ty);
                        return;
                    }
                };
                let (sig, map) = self.infcx.replace_late_bound_regions_with_fresh_var(
                    term.source_info.span,
                    LateBoundRegionConversionTime::FnCall,
                    &sig,
                );
                let sig = self.normalize(&sig, term_location);
                self.check_call_dest(mir, term, &sig, destination, term_location);

                // The ordinary liveness rules will ensure that all
                // regions in the type of the callee are live here. We
                // then further constrain the late-bound regions that
                // were instantiated at the call site to be live as
                // well. The resulting is that all the input (and
                // output) types in the signature must be live, since
                // all the inputs that fed into it were live.
                for &late_bound_region in map.values() {
                    self.constraints.liveness_set.push((
                        late_bound_region,
                        term_location,
                        Cause::LiveOther(term_location),
                    ));
                }

                if self.is_box_free(func) {
                    self.check_box_free_inputs(mir, term, &sig, args, term_location);
                } else {
                    self.check_call_inputs(mir, term, &sig, args, term_location);
                }
            }
            TerminatorKind::Assert {
                ref cond, ref msg, ..
            } => {
                let cond_ty = cond.ty(mir, tcx);
                if cond_ty != tcx.types.bool {
                    span_mirbug!(self, term, "bad Assert ({:?}, not bool", cond_ty);
                }

                if let AssertMessage::BoundsCheck { ref len, ref index } = *msg {
                    if len.ty(mir, tcx) != tcx.types.usize {
                        span_mirbug!(self, len, "bounds-check length non-usize {:?}", len)
                    }
                    if index.ty(mir, tcx) != tcx.types.usize {
                        span_mirbug!(self, index, "bounds-check index non-usize {:?}", index)
                    }
                }
            }
            TerminatorKind::Yield { ref value, .. } => {
                let value_ty = value.ty(mir, tcx);
                match mir.yield_ty {
                    None => span_mirbug!(self, term, "yield in non-generator"),
                    Some(ty) => {
                        if let Err(terr) = self.sub_types(value_ty, ty, term_location.at_self()) {
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
        mir: &Mir<'tcx>,
        term: &Terminator<'tcx>,
        sig: &ty::FnSig<'tcx>,
        destination: &Option<(Place<'tcx>, BasicBlock)>,
        term_location: Location,
    ) {
        let tcx = self.tcx();
        match *destination {
            Some((ref dest, target_block)) => {
                let dest_ty = dest.ty(mir, tcx).to_ty(tcx);
                let locations = Locations {
                    from_location: term_location,
                    at_location: target_block.start_location(),
                };
                if let Err(terr) = self.sub_types(sig.output(), dest_ty, locations) {
                    span_mirbug!(
                        self,
                        term,
                        "call dest mismatch ({:?} <- {:?}): {:?}",
                        dest_ty,
                        sig.output(),
                        terr
                    );
                }
            }
            None => {
                // FIXME(canndrew): This is_never should probably be an is_uninhabited
                if !sig.output().is_never() {
                    span_mirbug!(self, term, "call to converging function {:?} w/o dest", sig);
                }
            }
        }
    }

    fn check_call_inputs(
        &mut self,
        mir: &Mir<'tcx>,
        term: &Terminator<'tcx>,
        sig: &ty::FnSig<'tcx>,
        args: &[Operand<'tcx>],
        term_location: Location,
    ) {
        debug!("check_call_inputs({:?}, {:?})", sig, args);
        if args.len() < sig.inputs().len() || (args.len() > sig.inputs().len() && !sig.variadic) {
            span_mirbug!(self, term, "call to {:?} with wrong # of args", sig);
        }
        for (n, (fn_arg, op_arg)) in sig.inputs().iter().zip(args).enumerate() {
            let op_arg_ty = op_arg.ty(mir, self.tcx());
            if let Err(terr) = self.sub_types(op_arg_ty, fn_arg, term_location.at_self()) {
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

    fn is_box_free(&self, operand: &Operand<'tcx>) -> bool {
        match operand {
            &Operand::Constant(box Constant {
                literal:
                    Literal::Value {
                        value:
                            &ty::Const {
                                val: ConstVal::Function(def_id, _),
                                ..
                            },
                        ..
                    },
                ..
            }) => Some(def_id) == self.tcx().lang_items().box_free_fn(),
            _ => false,
        }
    }

    fn check_box_free_inputs(
        &mut self,
        mir: &Mir<'tcx>,
        term: &Terminator<'tcx>,
        sig: &ty::FnSig<'tcx>,
        args: &[Operand<'tcx>],
        term_location: Location,
    ) {
        debug!("check_box_free_inputs");

        // box_free takes a Box as a pointer. Allow for that.

        if sig.inputs().len() != 1 {
            span_mirbug!(self, term, "box_free should take 1 argument");
            return;
        }

        let pointee_ty = match sig.inputs()[0].sty {
            ty::TyRawPtr(mt) => mt.ty,
            _ => {
                span_mirbug!(self, term, "box_free should take a raw ptr");
                return;
            }
        };

        if args.len() != 1 {
            span_mirbug!(self, term, "box_free called with wrong # of args");
            return;
        }

        let ty = args[0].ty(mir, self.tcx());
        let arg_ty = match ty.sty {
            ty::TyRawPtr(mt) => mt.ty,
            ty::TyAdt(def, _) if def.is_box() => ty.boxed_ty(),
            _ => {
                span_mirbug!(self, term, "box_free called with bad arg ty");
                return;
            }
        };

        if let Err(terr) = self.sub_types(arg_ty, pointee_ty, term_location.at_self()) {
            span_mirbug!(
                self,
                term,
                "bad box_free arg ({:?} <- {:?}): {:?}",
                pointee_ty,
                arg_ty,
                terr
            );
        }
    }

    fn check_iscleanup(&mut self, mir: &Mir<'tcx>, block_data: &BasicBlockData<'tcx>) {
        let is_cleanup = block_data.is_cleanup;
        self.last_span = block_data.terminator().source_info.span;
        match block_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                self.assert_iscleanup(mir, block_data, target, is_cleanup)
            }
            TerminatorKind::SwitchInt { ref targets, .. } => for target in targets {
                self.assert_iscleanup(mir, block_data, *target, is_cleanup);
            },
            TerminatorKind::Resume => if !is_cleanup {
                span_mirbug!(self, block_data, "resume on non-cleanup block!")
            },
            TerminatorKind::Abort => if !is_cleanup {
                span_mirbug!(self, block_data, "abort on non-cleanup block!")
            },
            TerminatorKind::Return => if is_cleanup {
                span_mirbug!(self, block_data, "return on cleanup block")
            },
            TerminatorKind::GeneratorDrop { .. } => if is_cleanup {
                span_mirbug!(self, block_data, "generator_drop in cleanup block")
            },
            TerminatorKind::Yield { resume, drop, .. } => {
                if is_cleanup {
                    span_mirbug!(self, block_data, "yield in cleanup block")
                }
                self.assert_iscleanup(mir, block_data, resume, is_cleanup);
                if let Some(drop) = drop {
                    self.assert_iscleanup(mir, block_data, drop, is_cleanup);
                }
            }
            TerminatorKind::Unreachable => {}
            TerminatorKind::Drop { target, unwind, .. }
            | TerminatorKind::DropAndReplace { target, unwind, .. }
            | TerminatorKind::Assert {
                target,
                cleanup: unwind,
                ..
            } => {
                self.assert_iscleanup(mir, block_data, target, is_cleanup);
                if let Some(unwind) = unwind {
                    if is_cleanup {
                        span_mirbug!(self, block_data, "unwind on cleanup block")
                    }
                    self.assert_iscleanup(mir, block_data, unwind, true);
                }
            }
            TerminatorKind::Call {
                ref destination,
                cleanup,
                ..
            } => {
                if let &Some((_, target)) = destination {
                    self.assert_iscleanup(mir, block_data, target, is_cleanup);
                }
                if let Some(cleanup) = cleanup {
                    if is_cleanup {
                        span_mirbug!(self, block_data, "cleanup on cleanup block")
                    }
                    self.assert_iscleanup(mir, block_data, cleanup, true);
                }
            }
            TerminatorKind::FalseEdges {
                real_target,
                ref imaginary_targets,
            } => {
                self.assert_iscleanup(mir, block_data, real_target, is_cleanup);
                for target in imaginary_targets {
                    self.assert_iscleanup(mir, block_data, *target, is_cleanup);
                }
            }
        }
    }

    fn assert_iscleanup(
        &mut self,
        mir: &Mir<'tcx>,
        ctxt: &fmt::Debug,
        bb: BasicBlock,
        iscleanuppad: bool,
    ) {
        if mir[bb].is_cleanup != iscleanuppad {
            span_mirbug!(
                self,
                ctxt,
                "cleanuppad mismatch: {:?} should be {:?}",
                bb,
                iscleanuppad
            );
        }
    }

    fn check_local(&mut self, mir: &Mir<'tcx>, local: Local, local_decl: &LocalDecl<'tcx>) {
        match mir.local_kind(local) {
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

        let span = local_decl.source_info.span;
        let ty = local_decl.ty;

        // Erase the regions from `ty` to get a global type.  The
        // `Sized` bound in no way depends on precise regions, so this
        // shouldn't affect `is_sized`.
        let gcx = self.tcx().global_tcx();
        let erased_ty = gcx.lift(&self.tcx().erase_regions(&ty)).unwrap();
        if !erased_ty.is_sized(gcx, self.param_env, span) {
            // in current MIR construction, all non-control-flow rvalue
            // expressions evaluate through `as_temp` or `into` a return
            // slot or local, so to find all unsized rvalues it is enough
            // to check all temps, return slots and locals.
            if let None = self.reported_errors.replace((ty, span)) {
                span_err!(
                    self.tcx().sess,
                    span,
                    E0161,
                    "cannot move a value of type {0}: the size of {0} \
                     cannot be statically determined",
                    ty
                );
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
            AggregateKind::Adt(def, variant_index, substs, active_field_index) => {
                let variant = &def.variants[variant_index];
                let adj_field_index = active_field_index.unwrap_or(field_index);
                if let Some(field) = variant.fields.get(adj_field_index) {
                    Ok(self.normalize(&field.ty(tcx, substs), location))
                } else {
                    Err(FieldAccessError::OutOfRange {
                        field_count: variant.fields.len(),
                    })
                }
            }
            AggregateKind::Closure(def_id, substs) => {
                match substs.upvar_tys(def_id, tcx).nth(field_index) {
                    Some(ty) => Ok(ty),
                    None => Err(FieldAccessError::OutOfRange {
                        field_count: substs.upvar_tys(def_id, tcx).count(),
                    }),
                }
            }
            AggregateKind::Generator(def_id, substs, _) => {
                if let Some(ty) = substs.upvar_tys(def_id, tcx).nth(field_index) {
                    Ok(ty)
                } else {
                    match substs.field_tys(def_id, tcx).nth(field_index) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.field_tys(def_id, tcx).count() + 1,
                        }),
                    }
                }
            }
            AggregateKind::Array(ty) => Ok(ty),
            AggregateKind::Tuple => {
                unreachable!("This should have been covered in check_rvalues");
            }
        }
    }

    fn check_rvalue(&mut self, mir: &Mir<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx();

        match rvalue {
            Rvalue::Aggregate(ak, ops) => {
                self.check_aggregate_rvalue(mir, rvalue, ak, ops, location)
            }

            Rvalue::Repeat(operand, const_usize) => if const_usize.as_u64() > 1 {
                let operand_ty = operand.ty(mir, tcx);

                let trait_ref = ty::TraitRef {
                    def_id: tcx.lang_items().copy_trait().unwrap(),
                    substs: tcx.mk_substs_trait(operand_ty, &[]),
                };

                self.prove_trait_ref(trait_ref, location);
            },

            Rvalue::NullaryOp(_, ty) => {
                let trait_ref = ty::TraitRef {
                    def_id: tcx.lang_items().sized_trait().unwrap(),
                    substs: tcx.mk_substs_trait(ty, &[]),
                };

                self.prove_trait_ref(trait_ref, location);
            }

            Rvalue::Cast(cast_kind, op, ty) => match cast_kind {
                CastKind::ReifyFnPointer => {
                    let fn_sig = op.ty(mir, tcx).fn_sig(tcx);

                    // The type that we see in the fcx is like
                    // `foo::<'a, 'b>`, where `foo` is the path to a
                    // function definition. When we extract the
                    // signature, it comes from the `fn_sig` query,
                    // and hence may contain unnormalized results.
                    let fn_sig = self.normalize(&fn_sig, location);

                    let ty_fn_ptr_from = tcx.mk_fn_ptr(fn_sig);

                    if let Err(terr) = self.eq_types(ty_fn_ptr_from, ty, location.at_self()) {
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

                CastKind::ClosureFnPointer => {
                    let sig = match op.ty(mir, tcx).sty {
                        ty::TyClosure(def_id, substs) => {
                            substs.closure_sig_ty(def_id, tcx).fn_sig(tcx)
                        }
                        _ => bug!(),
                    };
                    let ty_fn_ptr_from = tcx.coerce_closure_fn_ty(sig);

                    if let Err(terr) = self.eq_types(ty_fn_ptr_from, ty, location.at_self()) {
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

                CastKind::UnsafeFnPointer => {
                    let fn_sig = op.ty(mir, tcx).fn_sig(tcx);

                    // The type that we see in the fcx is like
                    // `foo::<'a, 'b>`, where `foo` is the path to a
                    // function definition. When we extract the
                    // signature, it comes from the `fn_sig` query,
                    // and hence may contain unnormalized results.
                    let fn_sig = self.normalize(&fn_sig, location);

                    let ty_fn_ptr_from = tcx.safe_to_unsafe_fn_ty(fn_sig);

                    if let Err(terr) = self.eq_types(ty_fn_ptr_from, ty, location.at_self()) {
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

                CastKind::Unsize => {
                    let trait_ref = ty::TraitRef {
                        def_id: tcx.lang_items().coerce_unsized_trait().unwrap(),
                        substs: tcx.mk_substs_trait(op.ty(mir, tcx), &[ty]),
                    };

                    self.prove_trait_ref(trait_ref, location);
                }

                CastKind::Misc => {}
            },

            // FIXME: These other cases have to be implemented in future PRs
            Rvalue::Use(..)
            | Rvalue::Ref(..)
            | Rvalue::Len(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..) => {}
        }
    }

    fn check_aggregate_rvalue(
        &mut self,
        mir: &Mir<'tcx>,
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
            let operand_ty = operand.ty(mir, tcx);
            if let Err(terr) =
                self.sub_types(operand_ty, field_ty, location.at_successor_within_block())
            {
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

    fn prove_aggregate_predicates(
        &mut self,
        aggregate_kind: &AggregateKind<'tcx>,
        location: Location,
    ) {
        let tcx = self.tcx();

        debug!(
            "prove_aggregate_predicates(aggregate_kind={:?}, location={:?})",
            aggregate_kind,
            location
        );

        let instantiated_predicates = match aggregate_kind {
            AggregateKind::Adt(def, _, substs, _) => {
                tcx.predicates_of(def.did).instantiate(tcx, substs)
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
            AggregateKind::Closure(def_id, substs) => {
                if let Some(closure_region_requirements) = tcx.mir_borrowck(*def_id) {
                    closure_region_requirements.apply_requirements(
                        self.infcx,
                        self.body_id,
                        location,
                        *def_id,
                        *substs,
                    );
                }

                tcx.predicates_of(*def_id).instantiate(tcx, substs.substs)
            }

            AggregateKind::Generator(def_id, substs, _) => {
                tcx.predicates_of(*def_id).instantiate(tcx, substs.substs)
            }

            AggregateKind::Array(_) | AggregateKind::Tuple => ty::InstantiatedPredicates::empty(),
        };

        let predicates = self.normalize(&instantiated_predicates.predicates, location);
        debug!("prove_aggregate_predicates: predicates={:?}", predicates);
        self.prove_predicates(&predicates, location);
    }

    fn prove_trait_ref(&mut self, trait_ref: ty::TraitRef<'tcx>, location: Location) {
        self.prove_predicates(
            &[
                ty::Predicate::Trait(trait_ref.to_poly_trait_ref().to_poly_trait_predicate()),
            ],
            location,
        );
    }

    fn prove_predicates(&mut self, predicates: &[ty::Predicate<'tcx>], location: Location) {
        debug!(
            "prove_predicates(predicates={:?}, location={:?})",
            predicates,
            location
        );
        self.fully_perform_op(location.at_self(), |this| {
            let cause = this.misc(this.last_span);
            let obligations = predicates
                .iter()
                .map(|&p| traits::Obligation::new(cause.clone(), this.param_env, p))
                .collect();
            Ok(InferOk {
                value: (),
                obligations,
            })
        }).unwrap()
    }

    fn typeck_mir(&mut self, mir: &Mir<'tcx>) {
        self.last_span = mir.span;
        debug!("run_on_mir: {:?}", mir.span);

        for (local, local_decl) in mir.local_decls.iter_enumerated() {
            self.check_local(mir, local, local_decl);
        }

        for (block, block_data) in mir.basic_blocks().iter_enumerated() {
            let mut location = Location {
                block,
                statement_index: 0,
            };
            for stmt in &block_data.statements {
                if stmt.source_info.span != DUMMY_SP {
                    self.last_span = stmt.source_info.span;
                }
                self.check_stmt(mir, stmt, location);
                location.statement_index += 1;
            }

            self.check_terminator(mir, block_data.terminator(), location);
            self.check_iscleanup(mir, block_data);
        }
    }

    fn normalize<T>(&mut self, value: &T, location: Location) -> T
    where
        T: fmt::Debug + TypeFoldable<'tcx>,
    {
        self.fully_perform_op(location.at_self(), |this| {
            let mut selcx = traits::SelectionContext::new(this.infcx);
            let cause = this.misc(this.last_span);
            let traits::Normalized { value, obligations } =
                traits::normalize(&mut selcx, this.param_env, cause, value);
            Ok(InferOk { value, obligations })
        }).unwrap()
    }
}

pub struct TypeckMir;

impl MirPass for TypeckMir {
    fn run_pass<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource, mir: &mut Mir<'tcx>) {
        let def_id = src.def_id;
        let id = tcx.hir.as_local_node_id(def_id).unwrap();
        debug!("run_pass: {:?}", def_id);

        if tcx.sess.err_count() > 0 {
            // compiling a broken program can obviously result in a
            // broken MIR, so try not to report duplicate errors.
            return;
        }
        let param_env = tcx.param_env(def_id);
        tcx.infer_ctxt().enter(|infcx| {
            let _ = type_check_internal(&infcx, id, param_env, mir, &[], None, &mut |_| ());

            // For verification purposes, we just ignore the resulting
            // region constraint sets. Not our problem. =)
        });
    }
}

trait AtLocation {
    /// Creates a `Locations` where `self` is both the from-location
    /// and the at-location. This means that any required region
    /// relationships must hold upon entering the statement/terminator
    /// indicated by `self`. This is typically used when processing
    /// "inputs" to the given location.
    fn at_self(self) -> Locations;

    /// Creates a `Locations` where `self` is the from-location and
    /// its successor within the block is the at-location. This means
    /// that any required region relationships must hold only upon
    /// **exiting** the statement/terminator indicated by `self`. This
    /// is for example used when you have a `place = rv` statement: it
    /// indicates that the `typeof(rv) <: typeof(place)` as of the
    /// **next** statement.
    fn at_successor_within_block(self) -> Locations;
}

impl AtLocation for Location {
    fn at_self(self) -> Locations {
        Locations {
            from_location: self,
            at_location: self,
        }
    }

    fn at_successor_within_block(self) -> Locations {
        Locations {
            from_location: self,
            at_location: self.successor_within_block(),
        }
    }
}
