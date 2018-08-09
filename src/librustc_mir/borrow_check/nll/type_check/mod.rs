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

use borrow_check::borrow_set::BorrowSet;
use borrow_check::location::LocationTable;
use borrow_check::nll::constraints::{ConstraintSet, OutlivesConstraint};
use borrow_check::nll::facts::AllFacts;
use borrow_check::nll::region_infer::values::{LivenessValues, RegionValueElements};
use borrow_check::nll::region_infer::{ClosureRegionRequirementsExt, TypeTest};
use borrow_check::nll::type_check::free_region_relations::{CreateResult, UniversalRegionRelations};
use borrow_check::nll::type_check::liveness::liveness_map::NllLivenessMap;
use borrow_check::nll::universal_regions::UniversalRegions;
use borrow_check::nll::LocalWithRegion;
use borrow_check::nll::ToRegionVid;
use dataflow::move_paths::MoveData;
use dataflow::FlowAtLocation;
use dataflow::MaybeInitializedPlaces;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::infer::canonical::QueryRegionConstraint;
use rustc::infer::region_constraints::GenericKind;
use rustc::infer::{InferCtxt, LateBoundRegionConversionTime};
use rustc::mir::interpret::EvalErrorKind::BoundsCheck;
use rustc::mir::tcx::PlaceTy;
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::mir::*;
use rustc::traits::query::type_op;
use rustc::traits::query::{Fallible, NoSolution};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{self, CanonicalTy, RegionVid, ToPolyTraitRef, Ty, TyCtxt, TyKind};
use rustc_errors::Diagnostic;
use std::fmt;
use std::rc::Rc;
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
                $context.mir_def_id,
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

mod constraint_conversion;
pub mod free_region_relations;
mod input_output;
crate mod liveness;
mod relate_tys;

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
/// - `errors_buffer` -- errors are sent here for future reporting
pub(crate) fn type_check<'gcx, 'tcx>(
    infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'gcx>,
    mir: &Mir<'tcx>,
    mir_def_id: DefId,
    universal_regions: &Rc<UniversalRegions<'tcx>>,
    location_table: &LocationTable,
    borrow_set: &BorrowSet<'tcx>,
    all_facts: &mut Option<AllFacts>,
    flow_inits: &mut FlowAtLocation<MaybeInitializedPlaces<'_, 'gcx, 'tcx>>,
    move_data: &MoveData<'tcx>,
    elements: &Rc<RegionValueElements>,
    errors_buffer: &mut Vec<Diagnostic>,
) -> MirTypeckResults<'tcx> {
    let implicit_region_bound = infcx.tcx.mk_region(ty::ReVar(universal_regions.fr_fn_body));
    let mut constraints = MirTypeckRegionConstraints {
        liveness_constraints: LivenessValues::new(elements),
        outlives_constraints: ConstraintSet::default(),
        type_tests: Vec::default(),
    };

    let CreateResult {
        universal_region_relations,
        region_bound_pairs,
        normalized_inputs_and_output,
    } = free_region_relations::create(
        infcx,
        mir_def_id,
        param_env,
        location_table,
        Some(implicit_region_bound),
        universal_regions,
        &mut constraints,
        all_facts,
    );

    let (liveness, liveness_map) = {
        let mut borrowck_context = BorrowCheckContext {
            universal_regions,
            location_table,
            borrow_set,
            all_facts,
            constraints: &mut constraints,
        };

        type_check_internal(
            infcx,
            mir_def_id,
            param_env,
            mir,
            &region_bound_pairs,
            Some(implicit_region_bound),
            Some(&mut borrowck_context),
            Some(errors_buffer),
            |cx| {
                cx.equate_inputs_and_outputs(
                    mir,
                    mir_def_id,
                    universal_regions,
                    &universal_region_relations,
                    &normalized_inputs_and_output,
                );
                liveness::generate(cx, mir, flow_inits, move_data)
            },
        )
    };

    MirTypeckResults {
        constraints,
        universal_region_relations,
        liveness,
        liveness_map,
    }
}

fn type_check_internal<'a, 'gcx, 'tcx, R>(
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    mir_def_id: DefId,
    param_env: ty::ParamEnv<'gcx>,
    mir: &'a Mir<'tcx>,
    region_bound_pairs: &'a [(ty::Region<'tcx>, GenericKind<'tcx>)],
    implicit_region_bound: Option<ty::Region<'tcx>>,
    borrowck_context: Option<&'a mut BorrowCheckContext<'a, 'tcx>>,
    errors_buffer: Option<&mut Vec<Diagnostic>>,
    mut extra: impl FnMut(&mut TypeChecker<'a, 'gcx, 'tcx>) -> R,
) -> R where {
    let mut checker = TypeChecker::new(
        infcx,
        mir,
        mir_def_id,
        param_env,
        region_bound_pairs,
        implicit_region_bound,
        borrowck_context,
    );
    let errors_reported = {
        let mut verifier = TypeVerifier::new(&mut checker, mir);
        verifier.visit_mir(mir);
        verifier.errors_reported
    };

    if !errors_reported {
        // if verifier failed, don't do further checks to avoid ICEs
        checker.typeck_mir(mir, errors_buffer);
    }

    extra(&mut checker)
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
struct TypeVerifier<'a, 'b: 'a, 'gcx: 'tcx, 'tcx: 'b> {
    cx: &'a mut TypeChecker<'b, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    last_span: Span,
    mir_def_id: DefId,
    errors_reported: bool,
}

impl<'a, 'b, 'gcx, 'tcx> Visitor<'tcx> for TypeVerifier<'a, 'b, 'gcx, 'tcx> {
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
        self.sanitize_constant(constant, location);
        self.sanitize_type(constant, constant.ty);

        if let Some(user_ty) = constant.user_ty {
            if let Err(terr) =
                self.cx
                    .eq_canonical_type_and_type(user_ty, constant.ty, location.boring())
            {
                span_mirbug!(
                    self,
                    constant,
                    "bad constant user type {:?} vs {:?}: {:?}",
                    user_ty,
                    constant.ty,
                    terr,
                );
            }
        }
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
            mir_def_id: cx.mir_def_id,
            cx,
            last_span: mir.span,
            errors_reported: false,
        }
    }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.cx.infcx.tcx
    }

    fn sanitize_type(&mut self, parent: &dyn fmt::Debug, ty: Ty<'tcx>) -> Ty<'tcx> {
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
            constant, location
        );

        // FIXME(#46702) -- We need some way to get the predicates
        // associated with the "pre-evaluated" form of the
        // constant. For example, consider that the constant
        // may have associated constant projections (`<Foo as
        // Trait<'a, 'b>>::SOME_CONST`) that impose
        // constraints on `'a` and `'b`. These constraints
        // would be lost if we just look at the normalized
        // value.
        if let ty::FnDef(def_id, substs) = constant.literal.ty.sty {
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

            let instantiated_predicates = tcx.predicates_of(def_id).instantiate(tcx, substs);
            type_checker.normalize_and_prove_instantiated_predicates(
                instantiated_predicates,
                location.boring(),
            );
        }

        debug!("sanitize_constant: expected_ty={:?}", constant.literal.ty);

        if let Err(terr) = self.cx
            .eq_types(constant.literal.ty, constant.ty, location.boring())
        {
            span_mirbug!(
                self,
                constant,
                "constant {:?} should have type {:?} but has {:?} ({:?})",
                constant,
                constant.literal.ty,
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
            Place::Promoted(box (_index, sty)) => {
                let sty = self.sanitize_type(place, sty);
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
                PlaceTy::Ty { ty: sty }
            }
            Place::Static(box Static { def_id, ty: sty }) => {
                let sty = self.sanitize_type(place, sty);
                let ty = self.tcx().type_of(def_id);
                let ty = self.cx.normalize(ty, location);
                if let Err(terr) = self.cx.eq_types(ty, sty, location.boring()) {
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
            let tcx = self.tcx();
            let trait_ref = ty::TraitRef {
                def_id: tcx.lang_items().copy_trait().unwrap(),
                substs: tcx.mk_substs_trait(place_ty.to_ty(tcx), &[]),
            };

            // In order to have a Copy operand, the type T of the value must be Copy. Note that we
            // prove that T: Copy, rather than using the type_moves_by_default test. This is
            // important because type_moves_by_default ignores the resulting region obligations and
            // assumes they pass. This can result in bounds from Copy impls being unsoundly ignored
            // (e.g., #29149). Note that we decide to use Copy before knowing whether the bounds
            // fully apply: in effect, the rule is that if a value of some type could implement
            // Copy, then it must.
            self.cx.prove_trait_ref(trait_ref, location.interesting());
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
                let deref_ty = base_ty.builtin_deref(true);
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
                    ty::Array(inner, size) => {
                        let size = size.unwrap_usize(tcx);
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
                    ty::Slice(..) => base_ty,
                    _ => span_mirbug_and_err!(self, place, "slice of non-array {:?}", base_ty),
                },
            },
            ProjectionElem::Downcast(adt_def1, index) => match base_ty.sty {
                ty::Adt(adt_def, substs) if adt_def.is_enum() && adt_def == adt_def1 => {
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
                    Ok(ty) => if let Err(terr) = self.cx.eq_types(ty, fty, location.boring()) {
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
        parent: &dyn fmt::Debug,
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
                ty::Adt(adt_def, substs) if !adt_def.is_enum() => (&adt_def.variants[0], substs),
                ty::Closure(def_id, substs) => {
                    return match substs.upvar_tys(def_id, tcx).nth(field.index()) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.upvar_tys(def_id, tcx).count(),
                        }),
                    }
                }
                ty::Generator(def_id, substs, _) => {
                    // Try pre-transform fields first (upvars and current state)
                    if let Some(ty) = substs.pre_transforms_tys(def_id, tcx).nth(field.index()) {
                        return Ok(ty);
                    }

                    // Then try `field_tys` which contains all the fields, but it
                    // requires the final optimized MIR.
                    return match substs.field_tys(def_id, tcx).nth(field.index()) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.field_tys(def_id, tcx).count(),
                        }),
                    };
                }
                ty::Tuple(tys) => {
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
struct TypeChecker<'a, 'gcx: 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    param_env: ty::ParamEnv<'gcx>,
    last_span: Span,
    mir: &'a Mir<'tcx>,
    mir_def_id: DefId,
    region_bound_pairs: &'a [(ty::Region<'tcx>, GenericKind<'tcx>)],
    implicit_region_bound: Option<ty::Region<'tcx>>,
    reported_errors: FxHashSet<(Ty<'tcx>, Span)>,
    borrowck_context: Option<&'a mut BorrowCheckContext<'a, 'tcx>>,
}

struct BorrowCheckContext<'a, 'tcx: 'a> {
    universal_regions: &'a UniversalRegions<'tcx>,
    location_table: &'a LocationTable,
    all_facts: &'a mut Option<AllFacts>,
    borrow_set: &'a BorrowSet<'tcx>,
    constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
}

crate struct MirTypeckResults<'tcx> {
    crate constraints: MirTypeckRegionConstraints<'tcx>,
    crate universal_region_relations: Rc<UniversalRegionRelations<'tcx>>,
    crate liveness: LivenessResults<LocalWithRegion>,
    crate liveness_map: NllLivenessMap,
}

/// A collection of region constraints that must be satisfied for the
/// program to be considered well-typed.
crate struct MirTypeckRegionConstraints<'tcx> {
    /// In general, the type-checker is not responsible for enforcing
    /// liveness constraints; this job falls to the region inferencer,
    /// which performs a liveness analysis. However, in some limited
    /// cases, the MIR type-checker creates temporary regions that do
    /// not otherwise appear in the MIR -- in particular, the
    /// late-bound regions that it instantiates at call-sites -- and
    /// hence it must report on their liveness constraints.
    crate liveness_constraints: LivenessValues<RegionVid>,

    crate outlives_constraints: ConstraintSet,

    crate type_tests: Vec<TypeTest<'tcx>>,
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
    /// `'1` -- as a universal region -- is live everywhere).  In the
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
    All,

    /// A "boring" constraint (caused by the given location) is one that
    /// the user probably doesn't want to see described in diagnostics,
    /// because it is kind of an artifact of the type system setup.
    ///
    /// Example: `x = Foo { field: y }` technically creates
    /// intermediate regions representing the "type of `Foo { field: y
    /// }`", and data flows from `y` into those variables, but they
    /// are not very interesting. The assignment into `x` on the other
    /// hand might be.
    Boring(Location),

    /// An *important* outlives constraint (caused by the given
    /// location) is one that would be useful to highlight in
    /// diagnostics, because it represents a point where references
    /// flow from one spot to another (e.g., `x = y`)
    Interesting(Location),
}

impl Locations {
    pub fn from_location(&self) -> Option<Location> {
        match self {
            Locations::All => None,
            Locations::Boring(from_location) | Locations::Interesting(from_location) => {
                Some(*from_location)
            }
        }
    }

    /// Gets a span representing the location.
    pub fn span(&self, mir: &Mir<'_>) -> Span {
        let span_location = match self {
            Locations::All => Location::START,
            Locations::Boring(l) | Locations::Interesting(l) => *l,
        };
        mir.source_info(span_location).span
    }
}

impl<'a, 'gcx, 'tcx> TypeChecker<'a, 'gcx, 'tcx> {
    fn new(
        infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
        mir: &'a Mir<'tcx>,
        mir_def_id: DefId,
        param_env: ty::ParamEnv<'gcx>,
        region_bound_pairs: &'a [(ty::Region<'tcx>, GenericKind<'tcx>)],
        implicit_region_bound: Option<ty::Region<'tcx>>,
        borrowck_context: Option<&'a mut BorrowCheckContext<'a, 'tcx>>,
    ) -> Self {
        TypeChecker {
            infcx,
            last_span: DUMMY_SP,
            mir,
            mir_def_id,
            param_env,
            region_bound_pairs,
            implicit_region_bound,
            borrowck_context,
            reported_errors: FxHashSet(),
        }
    }

    /// Given some operation `op` that manipulates types, proves
    /// predicates, or otherwise uses the inference context, executes
    /// `op` and then executes all the further obligations that `op`
    /// returns. This will yield a set of outlives constraints amongst
    /// regions which are extracted and stored as having occurred at
    /// `locations`.
    ///
    /// **Any `rustc::infer` operations that might generate region
    /// constraints should occur within this method so that those
    /// constraints can be properly localized!**
    fn fully_perform_op<R>(
        &mut self,
        locations: Locations,
        op: impl type_op::TypeOp<'gcx, 'tcx, Output = R>,
    ) -> Fallible<R> {
        let (r, opt_data) = op.fully_perform(self.infcx)?;

        if let Some(data) = &opt_data {
            self.push_region_constraints(locations, data);
        }

        Ok(r)
    }

    fn push_region_constraints(
        &mut self,
        locations: Locations,
        data: &[QueryRegionConstraint<'tcx>],
    ) {
        debug!(
            "push_region_constraints: constraints generated at {:?} are {:#?}",
            locations, data
        );

        if let Some(ref mut borrowck_context) = self.borrowck_context {
            constraint_conversion::ConstraintConversion::new(
                self.infcx.tcx,
                borrowck_context.universal_regions,
                borrowck_context.location_table,
                self.region_bound_pairs,
                self.implicit_region_bound,
                self.param_env,
                locations,
                &mut borrowck_context.constraints.outlives_constraints,
                &mut borrowck_context.constraints.type_tests,
                &mut borrowck_context.all_facts,
            ).convert_all(&data);
        }
    }

    fn sub_types(&mut self, sub: Ty<'tcx>, sup: Ty<'tcx>, locations: Locations) -> Fallible<()> {
        relate_tys::sub_types(
            self.infcx,
            sub,
            sup,
            locations,
            self.borrowck_context.as_mut().map(|x| &mut **x),
        )
    }

    fn eq_types(&mut self, a: Ty<'tcx>, b: Ty<'tcx>, locations: Locations) -> Fallible<()> {
        relate_tys::eq_types(
            self.infcx,
            a,
            b,
            locations,
            self.borrowck_context.as_mut().map(|x| &mut **x),
        )
    }

    fn eq_canonical_type_and_type(
        &mut self,
        a: CanonicalTy<'tcx>,
        b: Ty<'tcx>,
        locations: Locations,
    ) -> Fallible<()> {
        relate_tys::eq_canonical_type_and_type(
            self.infcx,
            a,
            b,
            locations,
            self.borrowck_context.as_mut().map(|x| &mut **x),
        )
    }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn check_stmt(&mut self, mir: &Mir<'tcx>, stmt: &Statement<'tcx>, location: Location) {
        debug!("check_stmt: {:?}", stmt);
        let tcx = self.tcx();
        match stmt.kind {
            StatementKind::Assign(ref place, ref rv) => {
                // Assignments to temporaries are not "interesting";
                // they are not caused by the user, but rather artifacts
                // of lowering. Assignments to other sorts of places *are* interesting
                // though.
                let is_temp = if let Place::Local(l) = *place {
                    l != RETURN_PLACE &&
                    !mir.local_decls[l].is_user_variable.is_some()
                } else {
                    false
                };

                let locations = if is_temp {
                    location.boring()
                } else {
                    location.interesting()
                };

                let place_ty = place.ty(mir, tcx).to_ty(tcx);
                let rv_ty = rv.ty(mir, tcx);
                if let Err(terr) = self.sub_types(rv_ty, place_ty, locations) {
                    span_mirbug!(
                        self,
                        stmt,
                        "bad assignment ({:?} = {:?}): {:?}",
                        place_ty,
                        rv_ty,
                        terr
                    );
                }

                if let Some(user_ty) = self.rvalue_user_ty(rv) {
                    if let Err(terr) = self.eq_canonical_type_and_type(
                        user_ty,
                        rv_ty,
                        location.boring(),
                    ) {
                        span_mirbug!(
                            self,
                            stmt,
                            "bad user type on rvalue ({:?} = {:?}): {:?}",
                            user_ty,
                            rv_ty,
                            terr
                        );
                    }
                }

                self.check_rvalue(mir, rv, location);
                if !self.tcx().features().unsized_locals {
                    let trait_ref = ty::TraitRef {
                        def_id: tcx.lang_items().sized_trait().unwrap(),
                        substs: tcx.mk_substs_trait(place_ty, &[]),
                    };
                    self.prove_trait_ref(trait_ref, location.interesting());
                }
            }
            StatementKind::SetDiscriminant {
                ref place,
                variant_index,
            } => {
                let place_type = place.ty(mir, tcx).to_ty(tcx);
                let adt = match place_type.sty {
                    TyKind::Adt(adt, _) if adt.is_enum() => adt,
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
            StatementKind::UserAssertTy(c_ty, local) => {
                let local_ty = mir.local_decls()[local].ty;
                if let Err(terr) = self.eq_canonical_type_and_type(c_ty, local_ty, Locations::All) {
                    span_mirbug!(
                        self,
                        stmt,
                        "bad type assert ({:?} = {:?}): {:?}",
                        c_ty,
                        local_ty,
                        terr
                    );
                }
            }
            StatementKind::ReadForMatch(_)
            | StatementKind::StorageLive(_)
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
        errors_buffer: &mut Option<&mut Vec<Diagnostic>>,
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
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                // no checks needed for these
            }

            TerminatorKind::DropAndReplace {
                ref location,
                ref value,
                target: _,
                unwind: _,
            } => {
                let place_ty = location.ty(mir, tcx).to_ty(tcx);
                let rv_ty = value.ty(mir, tcx);

                let locations = term_location.interesting();
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
            TerminatorKind::SwitchInt {
                ref discr,
                switch_ty,
                ..
            } => {
                let discr_ty = discr.ty(mir, tcx);
                if let Err(terr) = self.sub_types(discr_ty, switch_ty, term_location.boring()) {
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
                    ty::FnDef(..) | ty::FnPtr(_) => func_ty.fn_sig(tcx),
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
                let sig = self.normalize(sig, term_location);
                self.check_call_dest(mir, term, &sig, destination, term_location, errors_buffer);

                self.prove_predicates(
                    sig.inputs().iter().map(|ty| ty::Predicate::WellFormed(ty)),
                    term_location.boring(),
                );

                // The ordinary liveness rules will ensure that all
                // regions in the type of the callee are live here. We
                // then further constrain the late-bound regions that
                // were instantiated at the call site to be live as
                // well. The resulting is that all the input (and
                // output) types in the signature must be live, since
                // all the inputs that fed into it were live.
                for &late_bound_region in map.values() {
                    if let Some(ref mut borrowck_context) = self.borrowck_context {
                        let region_vid = borrowck_context
                            .universal_regions
                            .to_region_vid(late_bound_region);
                        borrowck_context
                            .constraints
                            .liveness_constraints
                            .add_element(region_vid, term_location);
                    }
                }

                self.check_call_inputs(mir, term, &sig, args, term_location);
            }
            TerminatorKind::Assert {
                ref cond, ref msg, ..
            } => {
                let cond_ty = cond.ty(mir, tcx);
                if cond_ty != tcx.types.bool {
                    span_mirbug!(self, term, "bad Assert ({:?}, not bool", cond_ty);
                }

                if let BoundsCheck { ref len, ref index } = *msg {
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
                        if let Err(terr) = self.sub_types(value_ty, ty, term_location.interesting())
                        {
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
        errors_buffer: &mut Option<&mut Vec<Diagnostic>>,
    ) {
        let tcx = self.tcx();
        match *destination {
            Some((ref dest, _target_block)) => {
                let dest_ty = dest.ty(mir, tcx).to_ty(tcx);
                let is_temp = if let Place::Local(l) = *dest {
                    l != RETURN_PLACE &&
                    !mir.local_decls[l].is_user_variable.is_some()
                } else {
                    false
                };

                let locations = if is_temp {
                    term_location.boring()
                } else {
                    term_location.interesting()
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

                // When `#![feature(unsized_locals)]` is not enabled,
                // this check is done at `check_local`.
                if self.tcx().features().unsized_locals {
                    let span = term.source_info.span;
                    self.ensure_place_sized(dest_ty, span, errors_buffer);
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
            if let Err(terr) = self.sub_types(op_arg_ty, fn_arg, term_location.interesting()) {
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
            TerminatorKind::FalseUnwind {
                real_target,
                unwind,
            } => {
                self.assert_iscleanup(mir, block_data, real_target, is_cleanup);
                if let Some(unwind) = unwind {
                    if is_cleanup {
                        span_mirbug!(
                            self,
                            block_data,
                            "cleanup in cleanup block via false unwind"
                        );
                    }
                    self.assert_iscleanup(mir, block_data, unwind, true);
                }
            }
        }
    }

    fn assert_iscleanup(
        &mut self,
        mir: &Mir<'tcx>,
        ctxt: &dyn fmt::Debug,
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

    fn check_local(
        &mut self,
        mir: &Mir<'tcx>,
        local: Local,
        local_decl: &LocalDecl<'tcx>,
        errors_buffer: &mut Option<&mut Vec<Diagnostic>>,
    ) {
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

        // When `#![feature(unsized_locals)]` is enabled, only function calls
        // are checked in `check_call_dest`.
        if !self.tcx().features().unsized_locals {
            let span = local_decl.source_info.span;
            let ty = local_decl.ty;
            self.ensure_place_sized(ty, span, errors_buffer);
        }
    }

    fn ensure_place_sized(&mut self,
                          ty: Ty<'tcx>,
                          span: Span,
                          errors_buffer: &mut Option<&mut Vec<Diagnostic>>) {
        let tcx = self.tcx();

        // Erase the regions from `ty` to get a global type.  The
        // `Sized` bound in no way depends on precise regions, so this
        // shouldn't affect `is_sized`.
        let gcx = tcx.global_tcx();
        let erased_ty = gcx.lift(&tcx.erase_regions(&ty)).unwrap();
        if !erased_ty.is_sized(gcx.at(span), self.param_env) {
            // in current MIR construction, all non-control-flow rvalue
            // expressions evaluate through `as_temp` or `into` a return
            // slot or local, so to find all unsized rvalues it is enough
            // to check all temps, return slots and locals.
            if let None = self.reported_errors.replace((ty, span)) {
                let mut diag = struct_span_err!(
                    self.tcx().sess,
                    span,
                    E0161,
                    "cannot move a value of type {0}: the size of {0} \
                     cannot be statically determined",
                    ty
                );
                if let Some(ref mut errors_buffer) = *errors_buffer {
                    diag.buffer(errors_buffer);
                } else {
                    // we're allowed to use emit() here because the
                    // NLL migration will be turned on (and thus
                    // errors will need to be buffered) *only if*
                    // errors_buffer is Some.
                    diag.emit();
                }
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
                // Try pre-transform fields first (upvars and current state)
                if let Some(ty) = substs.pre_transforms_tys(def_id, tcx).nth(field_index) {
                    Ok(ty)
                } else {
                    // Then try `field_tys` which contains all the fields, but it
                    // requires the final optimized MIR.
                    match substs.field_tys(def_id, tcx).nth(field_index) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.field_tys(def_id, tcx).count(),
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

            Rvalue::Repeat(operand, len) => if *len > 1 {
                let operand_ty = operand.ty(mir, tcx);

                let trait_ref = ty::TraitRef {
                    def_id: tcx.lang_items().copy_trait().unwrap(),
                    substs: tcx.mk_substs_trait(operand_ty, &[]),
                };

                self.prove_trait_ref(trait_ref, location.interesting());
            },

            Rvalue::NullaryOp(_, ty) => {
                let trait_ref = ty::TraitRef {
                    def_id: tcx.lang_items().sized_trait().unwrap(),
                    substs: tcx.mk_substs_trait(ty, &[]),
                };

                self.prove_trait_ref(trait_ref, location.interesting());
            }

            Rvalue::Cast(cast_kind, op, ty) => match cast_kind {
                CastKind::ReifyFnPointer => {
                    let fn_sig = op.ty(mir, tcx).fn_sig(tcx);

                    // The type that we see in the fcx is like
                    // `foo::<'a, 'b>`, where `foo` is the path to a
                    // function definition. When we extract the
                    // signature, it comes from the `fn_sig` query,
                    // and hence may contain unnormalized results.
                    let fn_sig = self.normalize(fn_sig, location);

                    let ty_fn_ptr_from = tcx.mk_fn_ptr(fn_sig);

                    if let Err(terr) = self.eq_types(ty_fn_ptr_from, ty, location.interesting()) {
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
                        ty::Closure(def_id, substs) => {
                            substs.closure_sig_ty(def_id, tcx).fn_sig(tcx)
                        }
                        _ => bug!(),
                    };
                    let ty_fn_ptr_from = tcx.coerce_closure_fn_ty(sig);

                    if let Err(terr) = self.eq_types(ty_fn_ptr_from, ty, location.interesting()) {
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
                    let fn_sig = self.normalize(fn_sig, location);

                    let ty_fn_ptr_from = tcx.safe_to_unsafe_fn_ty(fn_sig);

                    if let Err(terr) = self.eq_types(ty_fn_ptr_from, ty, location.interesting()) {
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
                    let &ty = ty;
                    let trait_ref = ty::TraitRef {
                        def_id: tcx.lang_items().coerce_unsized_trait().unwrap(),
                        substs: tcx.mk_substs_trait(op.ty(mir, tcx), &[ty.into()]),
                    };

                    self.prove_trait_ref(trait_ref, location.interesting());
                }

                CastKind::Misc => {}
            },

            Rvalue::Ref(region, _borrow_kind, borrowed_place) => {
                self.add_reborrow_constraint(location, region, borrowed_place);
            }

            // FIXME: These other cases have to be implemented in future PRs
            Rvalue::Use(..)
            | Rvalue::Len(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..) => {}
        }
    }

    /// If this rvalue supports a user-given type annotation, then
    /// extract and return it. This represents the final type of the
    /// rvalue and will be unified with the inferred type.
    fn rvalue_user_ty(
        &self,
        rvalue: &Rvalue<'tcx>,
    ) -> Option<CanonicalTy<'tcx>> {
        match rvalue {
            Rvalue::Use(_) |
            Rvalue::Repeat(..) |
            Rvalue::Ref(..) |
            Rvalue::Len(..) |
            Rvalue::Cast(..) |
            Rvalue::BinaryOp(..) |
            Rvalue::CheckedBinaryOp(..) |
            Rvalue::NullaryOp(..) |
            Rvalue::UnaryOp(..) |
            Rvalue::Discriminant(..) =>
                None,

            Rvalue::Aggregate(aggregate, _) => match **aggregate {
                AggregateKind::Adt(_, _, _, user_ty, _) => user_ty,
                AggregateKind::Array(_) => None,
                AggregateKind::Tuple => None,
                AggregateKind::Closure(_, _) => None,
                AggregateKind::Generator(_, _, _) => None,
            }
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

            if let Err(terr) = self.sub_types(operand_ty, field_ty, location.boring()) {
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

    /// Add the constraints that arise from a borrow expression `&'a P` at the location `L`.
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
        let BorrowCheckContext {
            borrow_set,
            location_table,
            all_facts,
            constraints,
            ..
        } = match self.borrowck_context {
            Some(ref mut borrowck_context) => borrowck_context,
            None => return,
        };

        // In Polonius mode, we also push a `borrow_region` fact
        // linking the loan to the region (in some cases, though,
        // there is no loan associated with this borrow expression --
        // that occurs when we are borrowing an unsafe place, for
        // example).
        if let Some(all_facts) = all_facts {
            if let Some(borrow_index) = borrow_set.location_map.get(&location) {
                let region_vid = borrow_region.to_region_vid();
                all_facts.borrow_region.push((
                    region_vid,
                    *borrow_index,
                    location_table.mid_index(location),
                ));
            }
        }

        // If we are reborrowing the referent of another reference, we
        // need to add outlives relationships. In a case like `&mut
        // *p`, where the `p` has type `&'b mut Foo`, for example, we
        // need to ensure that `'b: 'a`.

        let mut borrowed_place = borrowed_place;

        debug!(
            "add_reborrow_constraint({:?}, {:?}, {:?})",
            location, borrow_region, borrowed_place
        );
        while let Place::Projection(box PlaceProjection { base, elem }) = borrowed_place {
            debug!("add_reborrow_constraint - iteration {:?}", borrowed_place);

            match *elem {
                ProjectionElem::Deref => {
                    let tcx = self.infcx.tcx;
                    let base_ty = base.ty(self.mir, tcx).to_ty(tcx);

                    debug!("add_reborrow_constraint - base_ty = {:?}", base_ty);
                    match base_ty.sty {
                        ty::Ref(ref_region, _, mutbl) => {
                            constraints.outlives_constraints.push(OutlivesConstraint {
                                sup: ref_region.to_region_vid(),
                                sub: borrow_region.to_region_vid(),
                                locations: location.boring(),
                            });

                            if let Some(all_facts) = all_facts {
                                all_facts.outlives.push((
                                    ref_region.to_region_vid(),
                                    borrow_region.to_region_vid(),
                                    location_table.mid_index(location),
                                ));
                            }

                            match mutbl {
                                hir::Mutability::MutImmutable => {
                                    // Immutable reference. We don't need the base
                                    // to be valid for the entire lifetime of
                                    // the borrow.
                                    break;
                                }
                                hir::Mutability::MutMutable => {
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

            // The "propagate" case. We need to check that our base is valid
            // for the borrow's lifetime.
            borrowed_place = base;
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

        let instantiated_predicates = match aggregate_kind {
            AggregateKind::Adt(def, _, substs, _, _) => {
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
                if let Some(closure_region_requirements) =
                    tcx.mir_borrowck(*def_id).closure_requirements
                {
                    let closure_constraints = closure_region_requirements.apply_requirements(
                        self.infcx.tcx,
                        location,
                        *def_id,
                        *substs,
                    );

                    // Hmm, are these constraints *really* boring?
                    self.push_region_constraints(location.boring(), &closure_constraints);
                }

                tcx.predicates_of(*def_id).instantiate(tcx, substs.substs)
            }

            AggregateKind::Generator(def_id, substs, _) => {
                tcx.predicates_of(*def_id).instantiate(tcx, substs.substs)
            }

            AggregateKind::Array(_) | AggregateKind::Tuple => ty::InstantiatedPredicates::empty(),
        };

        self.normalize_and_prove_instantiated_predicates(
            instantiated_predicates,
            location.boring(),
        );
    }

    fn prove_trait_ref(&mut self, trait_ref: ty::TraitRef<'tcx>, locations: Locations) {
        self.prove_predicates(
            Some(ty::Predicate::Trait(
                trait_ref.to_poly_trait_ref().to_poly_trait_predicate(),
            )),
            locations,
        );
    }

    fn normalize_and_prove_instantiated_predicates(
        &mut self,
        instantiated_predicates: ty::InstantiatedPredicates<'tcx>,
        locations: Locations,
    ) {
        for predicate in instantiated_predicates.predicates {
            let predicate = self.normalize(predicate, locations);
            self.prove_predicate(predicate, locations);
        }
    }

    fn prove_predicates(
        &mut self,
        predicates: impl IntoIterator<Item = ty::Predicate<'tcx>>,
        locations: Locations,
    ) {
        for predicate in predicates {
            debug!(
                "prove_predicates(predicate={:?}, locations={:?})",
                predicate, locations,
            );

            self.prove_predicate(predicate, locations);
        }
    }

    fn prove_predicate(&mut self, predicate: ty::Predicate<'tcx>, locations: Locations) {
        debug!(
            "prove_predicate(predicate={:?}, location={:?})",
            predicate, locations,
        );

        let param_env = self.param_env;
        self.fully_perform_op(
            locations,
            param_env.and(type_op::prove_predicate::ProvePredicate::new(predicate)),
        ).unwrap_or_else(|NoSolution| {
            span_mirbug!(self, NoSolution, "could not prove {:?}", predicate);
        })
    }

    fn typeck_mir(&mut self, mir: &Mir<'tcx>, mut errors_buffer: Option<&mut Vec<Diagnostic>>) {
        self.last_span = mir.span;
        debug!("run_on_mir: {:?}", mir.span);

        for (local, local_decl) in mir.local_decls.iter_enumerated() {
            self.check_local(mir, local, local_decl, &mut errors_buffer);
        }

        for (block, block_data) in mir.basic_blocks().iter_enumerated() {
            let mut location = Location {
                block,
                statement_index: 0,
            };
            for stmt in &block_data.statements {
                if !stmt.source_info.span.is_dummy() {
                    self.last_span = stmt.source_info.span;
                }
                self.check_stmt(mir, stmt, location);
                location.statement_index += 1;
            }

            self.check_terminator(mir, block_data.terminator(), location, &mut errors_buffer);
            self.check_iscleanup(mir, block_data);
        }
    }

    fn normalize<T>(&mut self, value: T, location: impl NormalizeLocation) -> T
    where
        T: type_op::normalize::Normalizable<'gcx, 'tcx> + Copy,
    {
        debug!("normalize(value={:?}, location={:?})", value, location);
        let param_env = self.param_env;
        self.fully_perform_op(
            location.to_locations(),
            param_env.and(type_op::normalize::Normalize::new(value)),
        ).unwrap_or_else(|NoSolution| {
            span_mirbug!(self, NoSolution, "failed to normalize `{:?}`", value);
            value
        })
    }
}

pub struct TypeckMir;

impl MirPass for TypeckMir {
    fn run_pass<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource, mir: &mut Mir<'tcx>) {
        let def_id = src.def_id;
        debug!("run_pass: {:?}", def_id);

        // When NLL is enabled, the borrow checker runs the typeck
        // itself, so we don't need this MIR pass anymore.
        if tcx.use_mir_borrowck() {
            return;
        }

        if tcx.sess.err_count() > 0 {
            // compiling a broken program can obviously result in a
            // broken MIR, so try not to report duplicate errors.
            return;
        }

        if tcx.is_struct_constructor(def_id) {
            // We just assume that the automatically generated struct constructors are
            // correct. See the comment in the `mir_borrowck` implementation for an
            // explanation why we need this.
            return;
        }

        let param_env = tcx.param_env(def_id);
        tcx.infer_ctxt().enter(|infcx| {
            type_check_internal(
                &infcx,
                def_id,
                param_env,
                mir,
                &[],
                None,
                None,
                None,
                |_| (),
            );

            // For verification purposes, we just ignore the resulting
            // region constraint sets. Not our problem. =)
        });
    }
}

pub trait AtLocation {
    /// Indicates a "boring" constraint that the user probably
    /// woudln't want to see highlights.
    fn boring(self) -> Locations;

    /// Indicates an "interesting" edge, which is of significance only
    /// for diagnostics.
    fn interesting(self) -> Locations;
}

impl AtLocation for Location {
    fn boring(self) -> Locations {
        Locations::Boring(self)
    }

    fn interesting(self) -> Locations {
        Locations::Interesting(self)
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
        self.boring()
    }
}
