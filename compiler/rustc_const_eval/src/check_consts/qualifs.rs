//! Structural const qualification.
//!
//! See the `Qualif` trait for more info.

// FIXME(const_trait_impl): This API should be really reworked. It's dangerously general for
// having basically only two use-cases that act in different ways.

use rustc_errors::ErrorGuaranteed;
use rustc_hir::LangItem;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, AdtDef, Ty};
use rustc_middle::{bug, mir};
use rustc_trait_selection::traits::{Obligation, ObligationCause, ObligationCtxt};
use tracing::instrument;

use super::ConstCx;

pub fn in_any_value_of_ty<'tcx>(
    cx: &ConstCx<'_, 'tcx>,
    ty: Ty<'tcx>,
    tainted_by_errors: Option<ErrorGuaranteed>,
) -> ConstQualifs {
    ConstQualifs {
        has_mut_interior: HasMutInterior::in_any_value_of_ty(cx, ty),
        needs_drop: NeedsDrop::in_any_value_of_ty(cx, ty),
        needs_non_const_drop: NeedsNonConstDrop::in_any_value_of_ty(cx, ty),
        tainted_by_errors,
    }
}

/// A "qualif"(-ication) is a way to look for something "bad" in the MIR that would disqualify some
/// code for promotion or prevent it from evaluating at compile time.
///
/// Normally, we would determine what qualifications apply to each type and error when an illegal
/// operation is performed on such a type. However, this was found to be too imprecise, especially
/// in the presence of `enum`s. If only a single variant of an enum has a certain qualification, we
/// needn't reject code unless it actually constructs and operates on the qualified variant.
///
/// To accomplish this, const-checking and promotion use a value-based analysis (as opposed to a
/// type-based one). Qualifications propagate structurally across variables: If a local (or a
/// projection of a local) is assigned a qualified value, that local itself becomes qualified.
pub trait Qualif {
    /// The name of the file used to debug the dataflow analysis that computes this qualif.
    const ANALYSIS_NAME: &'static str;

    /// Whether this `Qualif` is cleared when a local is moved from.
    const IS_CLEARED_ON_MOVE: bool = false;

    /// Whether this `Qualif` might be evaluated after the promotion and can encounter a promoted.
    const ALLOW_PROMOTED: bool = false;

    /// Extracts the field of `ConstQualifs` that corresponds to this `Qualif`.
    fn in_qualifs(qualifs: &ConstQualifs) -> bool;

    /// Returns `true` if *any* value of the given type could possibly have this `Qualif`.
    ///
    /// This function determines `Qualif`s when we cannot do a value-based analysis. Since qualif
    /// propagation is context-insensitive, this includes function arguments and values returned
    /// from a call to another function.
    ///
    /// It also determines the `Qualif`s for primitive types.
    fn in_any_value_of_ty<'tcx>(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool;

    /// Returns `true` if the `Qualif` is structural in an ADT's fields, i.e. if we may
    /// recurse into an operand *value* to determine whether it has this `Qualif`.
    ///
    /// If this returns false, `in_any_value_of_ty` will be invoked to determine the
    /// final qualif for this ADT.
    fn is_structural_in_adt_value<'tcx>(cx: &ConstCx<'_, 'tcx>, adt: AdtDef<'tcx>) -> bool;
}

/// Constant containing interior mutability (`UnsafeCell<T>`).
/// This must be ruled out to make sure that evaluating the constant at compile-time
/// and at *any point* during the run-time would produce the same result. In particular,
/// promotion of temporaries must not change program behavior; if the promoted could be
/// written to, that would be a problem.
pub struct HasMutInterior;

impl Qualif for HasMutInterior {
    const ANALYSIS_NAME: &'static str = "flow_has_mut_interior";

    fn in_qualifs(qualifs: &ConstQualifs) -> bool {
        qualifs.has_mut_interior
    }

    fn in_any_value_of_ty<'tcx>(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        // Avoid selecting for simple cases, such as builtin types.
        if ty.is_trivially_freeze() {
            return false;
        }

        // Avoid selecting for `UnsafeCell` either.
        if ty.ty_adt_def().is_some_and(|adt| adt.is_unsafe_cell()) {
            return true;
        }

        // We do not use `ty.is_freeze` here, because that requires revealing opaque types, which
        // requires borrowck, which in turn will invoke mir_const_qualifs again, causing a cycle error.
        // Instead we invoke an obligation context manually, and provide the opaque type inference settings
        // that allow the trait solver to just error out instead of cycling.
        let freeze_def_id = cx.tcx.require_lang_item(LangItem::Freeze, cx.body.span);
        // FIXME(#132279): Once we've got a typing mode which reveals opaque types using the HIR
        // typeck results without causing query cycles, we should use this here instead of defining
        // opaque types.
        let typing_env = ty::TypingEnv {
            typing_mode: ty::TypingMode::analysis_in_body(
                cx.tcx,
                cx.body.source.def_id().expect_local(),
            ),
            param_env: cx.typing_env.param_env,
        };
        let (infcx, param_env) = cx.tcx.infer_ctxt().build_with_typing_env(typing_env);
        let ocx = ObligationCtxt::new(&infcx);
        let obligation = Obligation::new(
            cx.tcx,
            ObligationCause::dummy_with_span(cx.body.span),
            param_env,
            ty::TraitRef::new(cx.tcx, freeze_def_id, [ty::GenericArg::from(ty)]),
        );
        ocx.register_obligation(obligation);
        let errors = ocx.select_all_or_error();
        !errors.is_empty()
    }

    fn is_structural_in_adt_value<'tcx>(_cx: &ConstCx<'_, 'tcx>, adt: AdtDef<'tcx>) -> bool {
        // Exactly one type, `UnsafeCell`, has the `HasMutInterior` qualif inherently.
        // It arises structurally for all other types.
        !adt.is_unsafe_cell()
    }
}

/// Constant containing an ADT that implements `Drop`.
/// This must be ruled out because implicit promotion would remove side-effects
/// that occur as part of dropping that value. N.B., the implicit promotion has
/// to reject const Drop implementations because even if side-effects are ruled
/// out through other means, the execution of the drop could diverge.
pub struct NeedsDrop;

impl Qualif for NeedsDrop {
    const ANALYSIS_NAME: &'static str = "flow_needs_drop";
    const IS_CLEARED_ON_MOVE: bool = true;
    const ALLOW_PROMOTED: bool = true;

    fn in_qualifs(qualifs: &ConstQualifs) -> bool {
        qualifs.needs_drop
    }

    fn in_any_value_of_ty<'tcx>(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        ty.needs_drop(cx.tcx, cx.typing_env)
    }

    fn is_structural_in_adt_value<'tcx>(cx: &ConstCx<'_, 'tcx>, adt: AdtDef<'tcx>) -> bool {
        !adt.has_dtor(cx.tcx)
    }
}

/// Constant containing an ADT that implements non-const `Drop`.
/// This must be ruled out because we cannot run `Drop` during compile-time.
pub struct NeedsNonConstDrop;

impl Qualif for NeedsNonConstDrop {
    const ANALYSIS_NAME: &'static str = "flow_needs_nonconst_drop";
    const IS_CLEARED_ON_MOVE: bool = true;
    const ALLOW_PROMOTED: bool = true;

    fn in_qualifs(qualifs: &ConstQualifs) -> bool {
        qualifs.needs_non_const_drop
    }

    #[instrument(level = "trace", skip(cx), ret)]
    fn in_any_value_of_ty<'tcx>(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        // If this doesn't need drop at all, then don't select `~const Destruct`.
        if !ty.needs_drop(cx.tcx, cx.typing_env) {
            return false;
        }

        // We check that the type is `~const Destruct` since that will verify that
        // the type is both `~const Drop` (if a drop impl exists for the adt), *and*
        // that the components of this type are also `~const Destruct`. This
        // amounts to verifying that there are no values in this ADT that may have
        // a non-const drop.
        let destruct_def_id = cx.tcx.require_lang_item(LangItem::Destruct, cx.body.span);
        let (infcx, param_env) = cx.tcx.infer_ctxt().build_with_typing_env(cx.typing_env);
        let ocx = ObligationCtxt::new(&infcx);
        ocx.register_obligation(Obligation::new(
            cx.tcx,
            ObligationCause::misc(cx.body.span, cx.def_id()),
            param_env,
            ty::Binder::dummy(ty::TraitRef::new(cx.tcx, destruct_def_id, [ty]))
                .to_host_effect_clause(
                    cx.tcx,
                    match cx.const_kind() {
                        rustc_hir::ConstContext::ConstFn => ty::BoundConstness::Maybe,
                        rustc_hir::ConstContext::Static(_)
                        | rustc_hir::ConstContext::Const { .. } => ty::BoundConstness::Const,
                    },
                ),
        ));
        !ocx.select_all_or_error().is_empty()
    }

    fn is_structural_in_adt_value<'tcx>(cx: &ConstCx<'_, 'tcx>, adt: AdtDef<'tcx>) -> bool {
        // As soon as an ADT has a destructor, then the drop becomes non-structural
        // in its value since:
        // 1. The destructor may have `~const` bounds which are not present on the type.
        //   Someone needs to check that those are satisfied.
        //   While this could be instead satisfied by checking that the `~const Drop`
        //   impl holds (i.e. replicating part of the `in_any_value_of_ty` logic above),
        //   even in this case, we have another problem, which is,
        // 2. The destructor may *modify* the operand being dropped, so even if we
        //   did recurse on the components of the operand, we may not be even dropping
        //   the same values that were present before the custom destructor was invoked.
        !adt.has_dtor(cx.tcx)
    }
}

// FIXME: Use `mir::visit::Visitor` for the `in_*` functions if/when it supports early return.

/// Returns `true` if this `Rvalue` contains qualif `Q`.
pub fn in_rvalue<'tcx, Q, F>(
    cx: &ConstCx<'_, 'tcx>,
    in_local: &mut F,
    rvalue: &Rvalue<'tcx>,
) -> bool
where
    Q: Qualif,
    F: FnMut(Local) -> bool,
{
    match rvalue {
        Rvalue::ThreadLocalRef(_) | Rvalue::NullaryOp(..) => {
            Q::in_any_value_of_ty(cx, rvalue.ty(cx.body, cx.tcx))
        }

        Rvalue::Discriminant(place) | Rvalue::Len(place) => {
            in_place::<Q, _>(cx, in_local, place.as_ref())
        }

        Rvalue::CopyForDeref(place) => in_place::<Q, _>(cx, in_local, place.as_ref()),

        Rvalue::Use(operand)
        | Rvalue::Repeat(operand, _)
        | Rvalue::UnaryOp(_, operand)
        | Rvalue::Cast(_, operand, _)
        | Rvalue::ShallowInitBox(operand, _) => in_operand::<Q, _>(cx, in_local, operand),

        Rvalue::BinaryOp(_, box (lhs, rhs)) => {
            in_operand::<Q, _>(cx, in_local, lhs) || in_operand::<Q, _>(cx, in_local, rhs)
        }

        Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => {
            // Special-case reborrows to be more like a copy of the reference.
            if let Some((place_base, ProjectionElem::Deref)) = place.as_ref().last_projection() {
                let base_ty = place_base.ty(cx.body, cx.tcx).ty;
                if let ty::Ref(..) = base_ty.kind() {
                    return in_place::<Q, _>(cx, in_local, place_base);
                }
            }

            in_place::<Q, _>(cx, in_local, place.as_ref())
        }

        Rvalue::WrapUnsafeBinder(op, _) => in_operand::<Q, _>(cx, in_local, op),

        Rvalue::Aggregate(kind, operands) => {
            // Return early if we know that the struct or enum being constructed is always
            // qualified.
            if let AggregateKind::Adt(adt_did, ..) = **kind {
                let def = cx.tcx.adt_def(adt_did);
                // Don't do any value-based reasoning for unions.
                // Also, if the ADT is not structural in its fields,
                // then we cannot recurse on its fields. Instead,
                // we fall back to checking the qualif for *any* value
                // of the ADT.
                if def.is_union() || !Q::is_structural_in_adt_value(cx, def) {
                    return Q::in_any_value_of_ty(cx, rvalue.ty(cx.body, cx.tcx));
                }
            }

            // Otherwise, proceed structurally...
            operands.iter().any(|o| in_operand::<Q, _>(cx, in_local, o))
        }
    }
}

/// Returns `true` if this `Place` contains qualif `Q`.
pub fn in_place<'tcx, Q, F>(cx: &ConstCx<'_, 'tcx>, in_local: &mut F, place: PlaceRef<'tcx>) -> bool
where
    Q: Qualif,
    F: FnMut(Local) -> bool,
{
    let mut place = place;
    while let Some((place_base, elem)) = place.last_projection() {
        match elem {
            ProjectionElem::Index(index) if in_local(index) => return true,

            ProjectionElem::Deref
            | ProjectionElem::Subtype(_)
            | ProjectionElem::Field(_, _)
            | ProjectionElem::OpaqueCast(_)
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Downcast(_, _)
            | ProjectionElem::Index(_)
            | ProjectionElem::UnwrapUnsafeBinder(_) => {}
        }

        let base_ty = place_base.ty(cx.body, cx.tcx);
        let proj_ty = base_ty.projection_ty(cx.tcx, elem).ty;
        if !Q::in_any_value_of_ty(cx, proj_ty) {
            return false;
        }

        // `Deref` currently unconditionally "qualifies" if `in_any_value_of_ty` returns true,
        // i.e., we treat all qualifs as non-structural for deref projections. Generally,
        // we can say very little about `*ptr` even if we know that `ptr` satisfies all
        // sorts of properties.
        if matches!(elem, ProjectionElem::Deref) {
            // We have to assume that this qualifies.
            return true;
        }

        place = place_base;
    }

    assert!(place.projection.is_empty());
    in_local(place.local)
}

/// Returns `true` if this `Operand` contains qualif `Q`.
pub fn in_operand<'tcx, Q, F>(
    cx: &ConstCx<'_, 'tcx>,
    in_local: &mut F,
    operand: &Operand<'tcx>,
) -> bool
where
    Q: Qualif,
    F: FnMut(Local) -> bool,
{
    let constant = match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            return in_place::<Q, _>(cx, in_local, place.as_ref());
        }

        Operand::Constant(c) => c,
    };

    // Check the qualifs of the value of `const` items.
    let uneval = match constant.const_ {
        Const::Ty(_, ct)
            if matches!(
                ct.kind(),
                ty::ConstKind::Param(_) | ty::ConstKind::Error(_) | ty::ConstKind::Value(_)
            ) =>
        {
            None
        }
        Const::Ty(_, c) => {
            bug!("expected ConstKind::Param or ConstKind::Value here, found {:?}", c)
        }
        Const::Unevaluated(uv, _) => Some(uv),
        Const::Val(..) => None,
    };

    if let Some(mir::UnevaluatedConst { def, args: _, promoted }) = uneval {
        // Use qualifs of the type for the promoted. Promoteds in MIR body should be possible
        // only for `NeedsNonConstDrop` with precise drop checking. This is the only const
        // check performed after the promotion. Verify that with an assertion.
        assert!(promoted.is_none() || Q::ALLOW_PROMOTED);

        // Don't peek inside trait associated constants.
        if promoted.is_none() && cx.tcx.trait_of_item(def).is_none() {
            let qualifs = cx.tcx.at(constant.span).mir_const_qualif(def);

            if !Q::in_qualifs(&qualifs) {
                return false;
            }

            // Just in case the type is more specific than
            // the definition, e.g., impl associated const
            // with type parameters, take it into account.
        }
    }

    // Otherwise use the qualifs of the type.
    Q::in_any_value_of_ty(cx, constant.const_.ty())
}
