//! Structural const qualification.
//!
//! See the `Qualif` trait for more info.

use rustc_errors::ErrorGuaranteed;
use rustc_hir::LangItem;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::bug;
use rustc_middle::mir;
use rustc_middle::mir::*;
use rustc_middle::traits::BuiltinImplSource;
use rustc_middle::ty::{self, AdtDef, GenericArgsRef, Ty};
use rustc_trait_selection::traits::{
    ImplSource, Obligation, ObligationCause, ObligationCtxt, SelectionContext,
};
use tracing::{instrument, trace};

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

    /// Returns `true` if this `Qualif` is inherent to the given struct or enum.
    ///
    /// By default, `Qualif`s propagate into ADTs in a structural way: An ADT only becomes
    /// qualified if part of it is assigned a value with that `Qualif`. However, some ADTs *always*
    /// have a certain `Qualif`, regardless of whether their fields have it. For example, a type
    /// with a custom `Drop` impl is inherently `NeedsDrop`.
    ///
    /// Returning `true` for `in_adt_inherently` but `false` for `in_any_value_of_ty` is unsound.
    fn in_adt_inherently<'tcx>(
        cx: &ConstCx<'_, 'tcx>,
        adt: AdtDef<'tcx>,
        args: GenericArgsRef<'tcx>,
    ) -> bool;

    /// Returns `true` if this `Qualif` behaves sructurally for pointers and references:
    /// the pointer/reference qualifies if and only if the pointee qualifies.
    ///
    /// (This is currently `false` for all our instances, but that may change in the future. Also,
    /// by keeping it abstract, the handling of `Deref` in `in_place` becomes more clear.)
    fn deref_structural<'tcx>(cx: &ConstCx<'_, 'tcx>) -> bool;
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

        // We do not use `ty.is_freeze` here, because that requires revealing opaque types, which
        // requires borrowck, which in turn will invoke mir_const_qualifs again, causing a cycle error.
        // Instead we invoke an obligation context manually, and provide the opaque type inference settings
        // that allow the trait solver to just error out instead of cycling.
        let freeze_def_id = cx.tcx.require_lang_item(LangItem::Freeze, Some(cx.body.span));

        let obligation = Obligation::new(
            cx.tcx,
            ObligationCause::dummy_with_span(cx.body.span),
            cx.param_env,
            ty::TraitRef::new(cx.tcx, freeze_def_id, [ty::GenericArg::from(ty)]),
        );

        let infcx = cx
            .tcx
            .infer_ctxt()
            .with_opaque_type_inference(cx.body.source.def_id().expect_local())
            .build();
        let ocx = ObligationCtxt::new(&infcx);
        ocx.register_obligation(obligation);
        let errors = ocx.select_all_or_error();
        !errors.is_empty()
    }

    fn in_adt_inherently<'tcx>(
        _cx: &ConstCx<'_, 'tcx>,
        adt: AdtDef<'tcx>,
        _: GenericArgsRef<'tcx>,
    ) -> bool {
        // Exactly one type, `UnsafeCell`, has the `HasMutInterior` qualif inherently.
        // It arises structurally for all other types.
        adt.is_unsafe_cell()
    }

    fn deref_structural<'tcx>(_cx: &ConstCx<'_, 'tcx>) -> bool {
        false
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

    fn in_qualifs(qualifs: &ConstQualifs) -> bool {
        qualifs.needs_drop
    }

    fn in_any_value_of_ty<'tcx>(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        ty.needs_drop(cx.tcx, cx.param_env)
    }

    fn in_adt_inherently<'tcx>(
        cx: &ConstCx<'_, 'tcx>,
        adt: AdtDef<'tcx>,
        _: GenericArgsRef<'tcx>,
    ) -> bool {
        adt.has_dtor(cx.tcx)
    }

    fn deref_structural<'tcx>(_cx: &ConstCx<'_, 'tcx>) -> bool {
        false
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
        // Avoid selecting for simple cases, such as builtin types.
        if ty::util::is_trivially_const_drop(ty) {
            return false;
        }

        // FIXME(effects): If `destruct` is not a `const_trait`,
        // or effects are disabled in this crate, then give up.
        let destruct_def_id = cx.tcx.require_lang_item(LangItem::Destruct, Some(cx.body.span));
        if !cx.tcx.has_host_param(destruct_def_id) || !cx.tcx.features().effects {
            return NeedsDrop::in_any_value_of_ty(cx, ty);
        }

        let obligation = Obligation::new(
            cx.tcx,
            ObligationCause::dummy_with_span(cx.body.span),
            cx.param_env,
            ty::TraitRef::new(
                cx.tcx,
                destruct_def_id,
                [
                    ty::GenericArg::from(ty),
                    ty::GenericArg::from(cx.tcx.expected_host_effect_param_for_body(cx.def_id())),
                ],
            ),
        );

        let infcx = cx.tcx.infer_ctxt().build();
        let mut selcx = SelectionContext::new(&infcx);
        let Some(impl_src) = selcx.select(&obligation).ok().flatten() else {
            // If we couldn't select a const destruct candidate, then it's bad
            return true;
        };

        trace!(?impl_src);

        if !matches!(
            impl_src,
            ImplSource::Builtin(BuiltinImplSource::Misc, _) | ImplSource::Param(_)
        ) {
            // If our const destruct candidate is not ConstDestruct or implied by the param env,
            // then it's bad
            return true;
        }

        if impl_src.borrow_nested_obligations().is_empty() {
            return false;
        }

        // If we had any errors, then it's bad
        let ocx = ObligationCtxt::new(&infcx);
        ocx.register_obligations(impl_src.nested_obligations());
        let errors = ocx.select_all_or_error();
        !errors.is_empty()
    }

    fn in_adt_inherently<'tcx>(
        cx: &ConstCx<'_, 'tcx>,
        adt: AdtDef<'tcx>,
        _: GenericArgsRef<'tcx>,
    ) -> bool {
        adt.has_non_const_dtor(cx.tcx)
    }

    fn deref_structural<'tcx>(_cx: &ConstCx<'_, 'tcx>) -> bool {
        false
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

        Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
            // Special-case reborrows to be more like a copy of the reference.
            if let Some((place_base, ProjectionElem::Deref)) = place.as_ref().last_projection() {
                let base_ty = place_base.ty(cx.body, cx.tcx).ty;
                if let ty::Ref(..) = base_ty.kind() {
                    return in_place::<Q, _>(cx, in_local, place_base);
                }
            }

            in_place::<Q, _>(cx, in_local, place.as_ref())
        }

        Rvalue::Aggregate(kind, operands) => {
            // Return early if we know that the struct or enum being constructed is always
            // qualified.
            if let AggregateKind::Adt(adt_did, _, args, ..) = **kind {
                let def = cx.tcx.adt_def(adt_did);
                if Q::in_adt_inherently(cx, def, args) {
                    return true;
                }
                // Don't do any value-based reasoning for unions.
                if def.is_union() && Q::in_any_value_of_ty(cx, rvalue.ty(cx.body, cx.tcx)) {
                    return true;
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
            | ProjectionElem::Index(_) => {}
        }

        let base_ty = place_base.ty(cx.body, cx.tcx);
        let proj_ty = base_ty.projection_ty(cx.tcx, elem).ty;
        if !Q::in_any_value_of_ty(cx, proj_ty) {
            return false;
        }

        if matches!(elem, ProjectionElem::Deref) && !Q::deref_structural(cx) {
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
                ty::ConstKind::Param(_) | ty::ConstKind::Error(_) | ty::ConstKind::Value(_, _)
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
