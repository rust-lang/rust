//! Structural const qualification.
//!
//! See the `Qualif` trait for more info.

use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, subst::SubstsRef, AdtDef, Ty};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits::{
    self, ImplSource, Obligation, ObligationCause, SelectionContext,
};

use super::ConstCx;

pub fn in_any_value_of_ty(
    cx: &ConstCx<'_, 'tcx>,
    ty: Ty<'tcx>,
    error_occured: Option<ErrorReported>,
) -> ConstQualifs {
    ConstQualifs {
        has_mut_interior: HasMutInterior::in_any_value_of_ty(cx, ty),
        needs_drop: NeedsDrop::in_any_value_of_ty(cx, ty),
        needs_non_const_drop: NeedsNonConstDrop::in_any_value_of_ty(cx, ty),
        custom_eq: CustomEq::in_any_value_of_ty(cx, ty),
        error_occured,
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
    /// propagation is context-insenstive, this includes function arguments and values returned
    /// from a call to another function.
    ///
    /// It also determines the `Qualif`s for primitive types.
    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool;

    /// Returns `true` if this `Qualif` is inherent to the given struct or enum.
    ///
    /// By default, `Qualif`s propagate into ADTs in a structural way: An ADT only becomes
    /// qualified if part of it is assigned a value with that `Qualif`. However, some ADTs *always*
    /// have a certain `Qualif`, regardless of whether their fields have it. For example, a type
    /// with a custom `Drop` impl is inherently `NeedsDrop`.
    ///
    /// Returning `true` for `in_adt_inherently` but `false` for `in_any_value_of_ty` is unsound.
    fn in_adt_inherently(
        cx: &ConstCx<'_, 'tcx>,
        adt: &'tcx AdtDef,
        substs: SubstsRef<'tcx>,
    ) -> bool;
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

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        !ty.is_freeze(cx.tcx.at(DUMMY_SP), cx.param_env)
    }

    fn in_adt_inherently(cx: &ConstCx<'_, 'tcx>, adt: &'tcx AdtDef, _: SubstsRef<'tcx>) -> bool {
        // Exactly one type, `UnsafeCell`, has the `HasMutInterior` qualif inherently.
        // It arises structurally for all other types.
        Some(adt.did) == cx.tcx.lang_items().unsafe_cell_type()
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

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        ty.needs_drop(cx.tcx, cx.param_env)
    }

    fn in_adt_inherently(cx: &ConstCx<'_, 'tcx>, adt: &'tcx AdtDef, _: SubstsRef<'tcx>) -> bool {
        adt.has_dtor(cx.tcx)
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

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, mut ty: Ty<'tcx>) -> bool {
        // Avoid selecting for simple cases.
        match ty::util::needs_drop_components(ty, &cx.tcx.data_layout).as_deref() {
            Ok([]) => return false,
            Err(ty::util::AlwaysRequiresDrop) => return true,
            // If we've got a single component, select with that
            // to increase the chance that we hit the selection cache.
            Ok([t]) => ty = t,
            Ok([..]) => {}
        }

        let Some(drop_trait) = cx.tcx.lang_items().drop_trait() else {
            // there is no way to define a type that needs non-const drop
            // without having the lang item present.
            return false;
        };
        let trait_ref =
            ty::TraitRef { def_id: drop_trait, substs: cx.tcx.mk_substs_trait(ty, &[]) };
        let obligation = Obligation::new(
            ObligationCause::dummy(),
            cx.param_env,
            ty::Binder::dummy(ty::TraitPredicate {
                trait_ref,
                constness: ty::BoundConstness::ConstIfConst,
                polarity: ty::ImplPolarity::Positive,
            }),
        );

        let implsrc = cx.tcx.infer_ctxt().enter(|infcx| {
            let mut selcx = SelectionContext::with_constness(&infcx, hir::Constness::Const);
            selcx.select(&obligation)
        });
        !matches!(
            implsrc,
            Ok(Some(
                ImplSource::ConstDrop(_) | ImplSource::Param(_, ty::BoundConstness::ConstIfConst)
            ))
        )
    }

    fn in_adt_inherently(cx: &ConstCx<'_, 'tcx>, adt: &'tcx AdtDef, _: SubstsRef<'tcx>) -> bool {
        adt.has_non_const_dtor(cx.tcx)
    }
}

/// A constant that cannot be used as part of a pattern in a `match` expression.
pub struct CustomEq;

impl Qualif for CustomEq {
    const ANALYSIS_NAME: &'static str = "flow_custom_eq";

    fn in_qualifs(qualifs: &ConstQualifs) -> bool {
        qualifs.custom_eq
    }

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        // If *any* component of a composite data type does not implement `Structural{Partial,}Eq`,
        // we know that at least some values of that type are not structural-match. I say "some"
        // because that component may be part of an enum variant (e.g.,
        // `Option::<NonStructuralMatchTy>::Some`), in which case some values of this type may be
        // structural-match (`Option::None`).
        let id = cx.tcx.hir().local_def_id_to_hir_id(cx.def_id());
        traits::search_for_structural_match_violation(id, cx.body.span, cx.tcx, ty).is_some()
    }

    fn in_adt_inherently(
        cx: &ConstCx<'_, 'tcx>,
        adt: &'tcx AdtDef,
        substs: SubstsRef<'tcx>,
    ) -> bool {
        let ty = cx.tcx.mk_ty(ty::Adt(adt, substs));
        !ty.is_structural_eq_shallow(cx.tcx)
    }
}

// FIXME: Use `mir::visit::Visitor` for the `in_*` functions if/when it supports early return.

/// Returns `true` if this `Rvalue` contains qualif `Q`.
pub fn in_rvalue<Q, F>(cx: &ConstCx<'_, 'tcx>, in_local: &mut F, rvalue: &Rvalue<'tcx>) -> bool
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

        Rvalue::Use(operand)
        | Rvalue::Repeat(operand, _)
        | Rvalue::UnaryOp(_, operand)
        | Rvalue::Cast(_, operand, _)
        | Rvalue::ShallowInitBox(operand, _) => in_operand::<Q, _>(cx, in_local, operand),

        Rvalue::BinaryOp(_, box (lhs, rhs)) | Rvalue::CheckedBinaryOp(_, box (lhs, rhs)) => {
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
            if let AggregateKind::Adt(def, _, substs, ..) = **kind {
                if Q::in_adt_inherently(cx, def, substs) {
                    return true;
                }
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
pub fn in_place<Q, F>(cx: &ConstCx<'_, 'tcx>, in_local: &mut F, place: PlaceRef<'tcx>) -> bool
where
    Q: Qualif,
    F: FnMut(Local) -> bool,
{
    let mut place = place;
    while let Some((place_base, elem)) = place.last_projection() {
        match elem {
            ProjectionElem::Index(index) if in_local(index) => return true,

            ProjectionElem::Deref
            | ProjectionElem::Field(_, _)
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

        place = place_base;
    }

    assert!(place.projection.is_empty());
    in_local(place.local)
}

/// Returns `true` if this `Operand` contains qualif `Q`.
pub fn in_operand<Q, F>(cx: &ConstCx<'_, 'tcx>, in_local: &mut F, operand: &Operand<'tcx>) -> bool
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
    if let Some(ct) = constant.literal.const_for_ty() {
        if let ty::ConstKind::Unevaluated(ty::Unevaluated { def, substs_: _, promoted }) = ct.val {
            // Use qualifs of the type for the promoted. Promoteds in MIR body should be possible
            // only for `NeedsNonConstDrop` with precise drop checking. This is the only const
            // check performed after the promotion. Verify that with an assertion.
            assert!(promoted.is_none() || Q::ALLOW_PROMOTED);
            // Don't peek inside trait associated constants.
            if promoted.is_none() && cx.tcx.trait_of_item(def.did).is_none() {
                let qualifs = if let Some((did, param_did)) = def.as_const_arg() {
                    cx.tcx.at(constant.span).mir_const_qualif_const_arg((did, param_did))
                } else {
                    cx.tcx.at(constant.span).mir_const_qualif(def.did)
                };

                if !Q::in_qualifs(&qualifs) {
                    return false;
                }

                // Just in case the type is more specific than
                // the definition, e.g., impl associated const
                // with type parameters, take it into account.
            }
        }
    }
    // Otherwise use the qualifs of the type.
    Q::in_any_value_of_ty(cx, constant.literal.ty())
}
