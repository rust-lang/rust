//! Structural const qualification.
//!
//! See the `Qualif` trait for more info.

use rustc_errors::ErrorReported;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, subst::SubstsRef, AdtDef, Ty};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits;
use rustc_index::vec::IndexVec;
use rustc_index::bit_set::BitSet;
use crate::dataflow::{JoinSemiLattice, fmt::DebugWithContext};

use super::ConstCx;

pub fn in_any_value_of_ty(
    cx: &ConstCx<'_, 'tcx>,
    ty: Ty<'tcx>,
    error_occured: Option<ErrorReported>,
) -> ConstQualifs {
    ConstQualifs {
        has_mut_interior: HasMutInterior::in_any_value_of_ty(cx, ty).is_some(),
        needs_drop: NeedsDrop::in_any_value_of_ty(cx, ty).is_some(),
        custom_eq: CustomEq::in_any_value_of_ty(cx, ty).is_some(),
        error_occured,
    }
}

/// A "qualif"(-ication) is a way to look for something "bad" in the MIR that would disqualify some
/// code for promotion or prevent it from evaluating at compile time.
///
/// Normally, we would determine what qualifications apply to each type and error when an illegal
/// operation is performed on such a type. However, this was found to be too imprecise, especially
/// in the presence of `enum`s. If only a single variant of an enum has a certain qualification, we
/// needn't reject code unless it actually constructs and operates on the qualifed variant.
///
/// To accomplish this, const-checking and promotion use a value-based analysis (as opposed to a
/// type-based one). Qualifications propagate structurally across variables: If a local (or a
/// projection of a local) is assigned a qualifed value, that local itself becomes qualifed.
pub(crate) trait Qualif {
    /// The name of the file used to debug the dataflow analysis that computes this qualif.
    const ANALYSIS_NAME: &'static str;

    /// The dataflow result type. If it's just qualified/not qualified, then
    /// you can just use a `()` (most qualifs do that). But if you need more state, use a
    /// custom enum.
    type Result: SetChoice + std::fmt::Debug = ();
    type Set: QualifsPerLocal<Self::Result> = <Self::Result as SetChoice>::Set;

    /// Whether this `Qualif` is cleared when a local is moved from.
    const IS_CLEARED_ON_MOVE: bool = false;

    /// Extracts the field of `ConstQualifs` that corresponds to this `Qualif`.
    fn in_qualifs(qualifs: &ConstQualifs) -> Option<Self::Result>;

    /// Returns `true` if *any* value of the given type could possibly have this `Qualif`.
    ///
    /// This function determines `Qualif`s when we cannot do a value-based analysis. Since qualif
    /// propagation is context-insenstive, this includes function arguments and values returned
    /// from a call to another function.
    ///
    /// It also determines the `Qualif`s for primitive types.
    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> Option<Self::Result>;

    /// Sometimes const fn calls cannot possibly contain the qualif, so we can treat function
    /// calls special here.
    fn in_any_function_call(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>, _args: Option<Self::Result>) -> Option<Self::Result> {
        Self::in_any_value_of_ty(cx, ty)
    }

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
    ) -> Option<Self::Result>;

    fn in_value_behind_ref(qualif: Option<Self::Result>) -> Option<Self::Result> {
        qualif
    }
}

pub(crate) trait SetChoice: Sized + Clone + JoinSemiLattice {
    type Set: QualifsPerLocal<Self> = IndexVec<Local, Option<Self>>;
}

impl SetChoice for () {
    type Set = BitSet<Local>;
}

pub(crate) trait QualifsPerLocal<Value>: Sized + Clone + JoinSemiLattice {
    fn new_empty(n: usize) -> Self;
    fn insert(&mut self, local: Local, val: Value);
    fn remove(&mut self, local: Local);
    fn clear(&mut self);
    fn get(&self, local: Local) -> Option<Value>;
}

impl QualifsPerLocal<()> for BitSet<Local> {
    fn new_empty(n: usize) -> Self {
        BitSet::new_empty(n)
    }
    fn insert(&mut self, local: Local, _: ()) {
        BitSet::insert(self, local);
    }
    fn remove(&mut self, local: Local) {
        BitSet::remove(self, local);
    }
    fn clear(&mut self) {
        BitSet::clear(self)
    }
    fn get(&self, local: Local) -> Option<()> {
        self.contains(local).then_some(())
    }
}

impl<T: Clone + Eq + JoinSemiLattice> QualifsPerLocal<T> for IndexVec<Local, Option<T>> {
    fn new_empty(n: usize) -> Self {
        IndexVec::from_elem_n(None, n)
    }
    fn insert(&mut self, local: Local, val: T) {
        self[local].join(&Some(val));
    }
    fn remove(&mut self, local: Local) {
        self[local] = None;
    }
    fn clear(&mut self) {
        for elem in self.iter_mut() {
            *elem = None;
        }
    }
    fn get(&self, local: Local) -> Option<T> {
        self[local].clone()
    }
}

/// Constant containing interior mutability (`UnsafeCell<T>`).
/// This must be ruled out to make sure that evaluating the constant at compile-time
/// and at *any point* during the run-time would produce the same result. In particular,
/// promotion of temporaries must not change program behavior; if the promoted could be
/// written to, that would be a problem.
pub struct HasMutInterior;

impl Qualif for HasMutInterior {
    const ANALYSIS_NAME: &'static str = "flow_has_mut_interior";

    fn in_qualifs(qualifs: &ConstQualifs) -> Option<()> {
        qualifs.has_mut_interior.then_some(())
    }

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> Option<()> {
        (!ty.is_freeze(cx.tcx.at(DUMMY_SP), cx.param_env)).then_some(())
    }

    fn in_adt_inherently(cx: &ConstCx<'_, 'tcx>, adt: &'tcx AdtDef, _: SubstsRef<'tcx>) -> Option<()> {
        // Exactly one type, `UnsafeCell`, has the `HasMutInterior` qualif inherently.
        // It arises structurally for all other types.
        (Some(adt.did) == cx.tcx.lang_items().unsafe_cell_type()).then_some(())
    }
}

/// Constant containing interior mutability (`UnsafeCell<T>`) behind a reference.
/// This must be ruled out to make sure that evaluating the constant at compile-time
/// and at *any point* during the run-time would produce the same result. In particular,
/// promotion of temporaries must not change program behavior; if the promoted could be
/// written to, that would be a problem.
pub struct HasMutInteriorBehindRef;

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum HasMutInteriorBehindRefState {
    Yes,
    /// As long as we haven't encountered a reference yet, we use this state
    /// which is equivalent to the `HasMutInterior` qualif.
    OnlyHasMutInterior,
}
impl SetChoice for HasMutInteriorBehindRefState {}
impl<C> DebugWithContext<C> for HasMutInteriorBehindRefState {}

impl JoinSemiLattice for HasMutInteriorBehindRefState {
    fn join(&mut self, other: &Self) -> bool {
        match (&self, other) {
            (Self::Yes, _) => false,
            (Self::OnlyHasMutInterior, Self::Yes) => {
                *self = Self::Yes;
                true
            },
            (Self::OnlyHasMutInterior, Self::OnlyHasMutInterior) => false,
        }
    }
}

impl Qualif for HasMutInteriorBehindRef {
    const ANALYSIS_NAME: &'static str = "flow_has_mut_interior_behind_ref";
    type Result = HasMutInteriorBehindRefState;

    fn in_qualifs(qualifs: &ConstQualifs) -> Option<HasMutInteriorBehindRefState> {
        HasMutInterior::in_qualifs(qualifs).map(|()| HasMutInteriorBehindRefState::OnlyHasMutInterior)
    }

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> Option<HasMutInteriorBehindRefState> {
        match ty.builtin_deref(false) {
            None => HasMutInterior::in_any_value_of_ty(cx, ty).map(|()| HasMutInteriorBehindRefState::OnlyHasMutInterior),
            Some(tam) => HasMutInterior::in_any_value_of_ty(cx, tam.ty).map(|()| HasMutInteriorBehindRefState::Yes),
        }
    }

    #[instrument(skip(cx))]
    fn in_any_function_call(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>, mut args: Option<Self::Result>) -> Option<Self::Result> {
        args.join(&HasMutInterior::in_any_value_of_ty(cx, ty).map(|()| HasMutInteriorBehindRefState::OnlyHasMutInterior));
        args
    }

    fn in_adt_inherently(cx: &ConstCx<'_, 'tcx>, adt: &'tcx AdtDef, substs: SubstsRef<'tcx>) -> Option<HasMutInteriorBehindRefState> {
        HasMutInterior::in_adt_inherently(cx, adt, substs).map(|()| HasMutInteriorBehindRefState::OnlyHasMutInterior)
    }

    fn in_value_behind_ref(qualif: Option<HasMutInteriorBehindRefState>) -> Option<HasMutInteriorBehindRefState> {
        qualif.map(|_| HasMutInteriorBehindRefState::Yes)
    }
}

/// Constant containing an ADT that implements `Drop`.
/// This must be ruled out (a) because we cannot run `Drop` during compile-time
/// as that might not be a `const fn`, and (b) because implicit promotion would
/// remove side-effects that occur as part of dropping that value.
pub struct NeedsDrop;

impl Qualif for NeedsDrop {
    const ANALYSIS_NAME: &'static str = "flow_needs_drop";
    const IS_CLEARED_ON_MOVE: bool = true;

    fn in_qualifs(qualifs: &ConstQualifs) -> Option<()> {
        qualifs.needs_drop.then_some(())
    }

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> Option<()> {
        ty.needs_drop(cx.tcx, cx.param_env).then_some(())
    }

    fn in_adt_inherently(cx: &ConstCx<'_, 'tcx>, adt: &'tcx AdtDef, _: SubstsRef<'tcx>) -> Option<()> {
        adt.has_dtor(cx.tcx).then_some(())
    }
}

/// A constant that cannot be used as part of a pattern in a `match` expression.
pub struct CustomEq;

impl Qualif for CustomEq {
    const ANALYSIS_NAME: &'static str = "flow_custom_eq";

    fn in_qualifs(qualifs: &ConstQualifs) -> Option<()> {
        qualifs.custom_eq.then_some(())
    }

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> Option<()> {
        // If *any* component of a composite data type does not implement `Structural{Partial,}Eq`,
        // we know that at least some values of that type are not structural-match. I say "some"
        // because that component may be part of an enum variant (e.g.,
        // `Option::<NonStructuralMatchTy>::Some`), in which case some values of this type may be
        // structural-match (`Option::None`).
        let id = cx.tcx.hir().local_def_id_to_hir_id(cx.def_id());
        traits::search_for_structural_match_violation(id, cx.body.span, cx.tcx, ty).map(drop)
    }

    fn in_adt_inherently(
        cx: &ConstCx<'_, 'tcx>,
        adt: &'tcx AdtDef,
        substs: SubstsRef<'tcx>,
    ) -> Option<()> {
        let ty = cx.tcx.mk_ty(ty::Adt(adt, substs));
        (!ty.is_structural_eq_shallow(cx.tcx)).then_some(())
    }
}

// FIXME: Use `mir::visit::Visitor` for the `in_*` functions if/when it supports early return.

/// Returns `true` if this `Rvalue` contains qualif `Q`.
#[instrument(skip(cx, in_local), fields(Q=std::any::type_name::<Q>()))]
pub(crate) fn in_rvalue<Q, F>(cx: &ConstCx<'_, 'tcx>, in_local: &mut F, rvalue: &Rvalue<'tcx>) -> Option<Q::Result>
where
    Q: Qualif,
    F: FnMut(Local) -> Option<Q::Result>,
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
        | Rvalue::Cast(_, operand, _) => in_operand::<Q, _>(cx, in_local, operand),

        Rvalue::BinaryOp(_, lhs, rhs) | Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
            let mut res = in_operand::<Q, _>(cx, in_local, lhs);
            res.join(&in_operand::<Q, _>(cx, in_local, rhs));
            res
        }

        Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
            // Special-case reborrows to be more like a copy of the reference.
            if let &[ref proj_base @ .., ProjectionElem::Deref] = place.projection.as_ref() {
                let base_ty = Place::ty_from(place.local, proj_base, cx.body, cx.tcx).ty;
                if let ty::Ref(..) = base_ty.kind() {
                    return in_place::<Q, _>(
                        cx,
                        in_local,
                        PlaceRef { local: place.local, projection: proj_base },
                    );
                }
            }

            Q::in_value_behind_ref(in_place::<Q, _>(cx, in_local, place.as_ref()))
        }

        Rvalue::Aggregate(kind, operands) => {
            // Check if we know that the struct or enum being constructed is always qualified.
            let mut result = None;
            if let AggregateKind::Adt(def, _, substs, ..) = **kind {
                result.join(&Q::in_adt_inherently(cx, def, substs));
            }

            // Otherwise, proceed structurally...
            for o in operands {
                result.join(&in_operand::<Q, _>(cx, in_local, o));
            }
            result
        }
    }
}

/// Returns `true` if this `Place` contains qualif `Q`.
#[instrument(skip(cx, in_local), fields(Q=std::any::type_name::<Q>()))]
pub(crate) fn in_place<Q, F>(cx: &ConstCx<'_, 'tcx>, in_local: &mut F, place: PlaceRef<'tcx>) -> Option<Q::Result>
where
    Q: Qualif,
    F: FnMut(Local) -> Option<Q::Result>,
{
    let mut projection = place.projection;
    let mut result = None;
    while let &[ref proj_base @ .., proj_elem] = projection {
        match proj_elem {
            ProjectionElem::Index(index) => {
                result.join(&in_local(index));
            },

            ProjectionElem::Deref
            | ProjectionElem::Field(_, _)
            | ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Downcast(_, _) => {}
        }

        let base_ty = Place::ty_from(place.local, proj_base, cx.body, cx.tcx);
        let proj_ty = base_ty.projection_ty(cx.tcx, proj_elem).ty;
        if Q::in_any_value_of_ty(cx, proj_ty).is_none() {
            return result;
        }

        projection = proj_base;
    }

    assert!(projection.is_empty());
    result.join(&in_local(place.local));
    result
}

/// Returns `true` if this `Operand` contains qualif `Q`.
#[instrument(skip(cx, in_local), fields(Q=std::any::type_name::<Q>()))]
pub(crate) fn in_operand<Q, F>(cx: &ConstCx<'_, 'tcx>, in_local: &mut F, operand: &Operand<'tcx>) -> Option<Q::Result>
where
    Q: Qualif,
    F: FnMut(Local) -> Option<Q::Result>,
{
    let constant = match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            return in_place::<Q, _>(cx, in_local, place.as_ref());
        }

        Operand::Constant(c) => c,
    };

    // Check the qualifs of the value of `const` items.
    if let ty::ConstKind::Unevaluated(def, _, promoted) = constant.literal.val {
        assert!(promoted.is_none());
        // Don't peek inside trait associated constants.
        if cx.tcx.trait_of_item(def.did).is_none() {
            let qualifs = if let Some((did, param_did)) = def.as_const_arg() {
                cx.tcx.at(constant.span).mir_const_qualif_const_arg((did, param_did))
            } else {
                cx.tcx.at(constant.span).mir_const_qualif(def.did)
            };

            // Since this comes from a constant's qualifs, there can only
            // be `Option<()>` style qualifs, so we are allowed to early
            // return here and not try to join the results.
            Q::in_qualifs(&qualifs)?;

            // Just in case the type is more specific than
            // the definition, e.g., impl associated const
            // with type parameters, take it into account.
        }
    }
    // Otherwise use the qualifs of the type.
    Q::in_any_value_of_ty(cx, constant.literal.ty)
}
