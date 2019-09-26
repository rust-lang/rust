//! A copy of the `Qualif` trait in `qualify_consts.rs` that is suitable for the new validator.

use rustc::mir::*;
use rustc::mir::interpret::ConstValue;
use rustc::ty::{self, Ty};
use rustc_index::bit_set::BitSet;
use syntax_pos::DUMMY_SP;

use super::Item as ConstCx;
use super::validation::Mode;

#[derive(Clone, Copy)]
pub struct QualifSet(u8);

impl QualifSet {
    fn contains<Q: ?Sized + Qualif>(self) -> bool {
        self.0 & (1 << Q::IDX) != 0
    }
}

/// A "qualif"(-ication) is a way to look for something "bad" in the MIR that would disqualify some
/// code for promotion or prevent it from evaluating at compile time. So `return true` means
/// "I found something bad, no reason to go on searching". `false` is only returned if we
/// definitely cannot find anything bad anywhere.
///
/// The default implementations proceed structurally.
pub trait Qualif {
    const IDX: usize;

    /// Whether this `Qualif` is cleared when a local is moved from.
    const IS_CLEARED_ON_MOVE: bool = false;

    /// Return the qualification that is (conservatively) correct for any value
    /// of the type.
    fn in_any_value_of_ty(_cx: &ConstCx<'_, 'tcx>, _ty: Ty<'tcx>) -> bool;

    fn in_static(_cx: &ConstCx<'_, 'tcx>, _static: &Static<'tcx>) -> bool {
        // FIXME(eddyb) should we do anything here for value properties?
        false
    }

    fn in_projection_structurally(
        cx: &ConstCx<'_, 'tcx>,
        per_local: &BitSet<Local>,
        place: PlaceRef<'_, 'tcx>,
    ) -> bool {
        if let [proj_base @ .., elem] = place.projection {
            let base_qualif = Self::in_place(cx, per_local, PlaceRef {
                base: place.base,
                projection: proj_base,
            });
            let qualif = base_qualif && Self::in_any_value_of_ty(
                cx,
                Place::ty_from(place.base, proj_base, cx.body, cx.tcx)
                    .projection_ty(cx.tcx, elem)
                    .ty,
            );
            match elem {
                ProjectionElem::Deref |
                ProjectionElem::Subslice { .. } |
                ProjectionElem::Field(..) |
                ProjectionElem::ConstantIndex { .. } |
                ProjectionElem::Downcast(..) => qualif,

                ProjectionElem::Index(local) => qualif || per_local.contains(*local),
            }
        } else {
            bug!("This should be called if projection is not empty");
        }
    }

    fn in_projection(
        cx: &ConstCx<'_, 'tcx>,
        per_local: &BitSet<Local>,
        place: PlaceRef<'_, 'tcx>,
    ) -> bool {
        Self::in_projection_structurally(cx, per_local, place)
    }

    fn in_place(
        cx: &ConstCx<'_, 'tcx>,
        per_local: &BitSet<Local>,
        place: PlaceRef<'_, 'tcx>,
    ) -> bool {
        match place {
            PlaceRef {
                base: PlaceBase::Local(local),
                projection: [],
            } => per_local.contains(*local),
            PlaceRef {
                base: PlaceBase::Static(box Static {
                    kind: StaticKind::Promoted(..),
                    ..
                }),
                projection: [],
            } => bug!("qualifying already promoted MIR"),
            PlaceRef {
                base: PlaceBase::Static(static_),
                projection: [],
            } => {
                Self::in_static(cx, static_)
            },
            PlaceRef {
                base: _,
                projection: [.., _],
            } => Self::in_projection(cx, per_local, place),
        }
    }

    fn in_operand(
        cx: &ConstCx<'_, 'tcx>,
        per_local: &BitSet<Local>,
        operand: &Operand<'tcx>,
    ) -> bool {
        match *operand {
            Operand::Copy(ref place) |
            Operand::Move(ref place) => Self::in_place(cx, per_local, place.as_ref()),

            Operand::Constant(ref constant) => {
                if let ConstValue::Unevaluated(def_id, _) = constant.literal.val {
                    // Don't peek inside trait associated constants.
                    if cx.tcx.trait_of_item(def_id).is_some() {
                        Self::in_any_value_of_ty(cx, constant.literal.ty)
                    } else {
                        let (bits, _) = cx.tcx.at(constant.span).mir_const_qualif(def_id);

                        let qualif = QualifSet(bits).contains::<Self>();

                        // Just in case the type is more specific than
                        // the definition, e.g., impl associated const
                        // with type parameters, take it into account.
                        qualif && Self::in_any_value_of_ty(cx, constant.literal.ty)
                    }
                } else {
                    false
                }
            }
        }
    }

    fn in_rvalue_structurally(
        cx: &ConstCx<'_, 'tcx>,
        per_local: &BitSet<Local>,
        rvalue: &Rvalue<'tcx>,
    ) -> bool {
        match *rvalue {
            Rvalue::NullaryOp(..) => false,

            Rvalue::Discriminant(ref place) |
            Rvalue::Len(ref place) => Self::in_place(cx, per_local, place.as_ref()),

            Rvalue::Use(ref operand) |
            Rvalue::Repeat(ref operand, _) |
            Rvalue::UnaryOp(_, ref operand) |
            Rvalue::Cast(_, ref operand, _) => Self::in_operand(cx, per_local, operand),

            Rvalue::BinaryOp(_, ref lhs, ref rhs) |
            Rvalue::CheckedBinaryOp(_, ref lhs, ref rhs) => {
                Self::in_operand(cx, per_local, lhs) || Self::in_operand(cx, per_local, rhs)
            }

            Rvalue::Ref(_, _, ref place) => {
                // Special-case reborrows to be more like a copy of the reference.
                if let box [proj_base @ .., elem] = &place.projection {
                    if ProjectionElem::Deref == *elem {
                        let base_ty = Place::ty_from(&place.base, proj_base, cx.body, cx.tcx).ty;
                        if let ty::Ref(..) = base_ty.kind {
                            return Self::in_place(cx, per_local, PlaceRef {
                                base: &place.base,
                                projection: proj_base,
                            });
                        }
                    }
                }

                Self::in_place(cx, per_local, place.as_ref())
            }

            Rvalue::Aggregate(_, ref operands) => {
                operands.iter().any(|o| Self::in_operand(cx, per_local, o))
            }
        }
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, per_local: &BitSet<Local>, rvalue: &Rvalue<'tcx>) -> bool {
        Self::in_rvalue_structurally(cx, per_local, rvalue)
    }

    fn in_call(
        cx: &ConstCx<'_, 'tcx>,
        _per_local: &BitSet<Local>,
        _callee: &Operand<'tcx>,
        _args: &[Operand<'tcx>],
        return_ty: Ty<'tcx>,
    ) -> bool {
        // Be conservative about the returned value of a const fn.
        Self::in_any_value_of_ty(cx, return_ty)
    }
}

/// Constant containing interior mutability (`UnsafeCell<T>`).
/// This must be ruled out to make sure that evaluating the constant at compile-time
/// and at *any point* during the run-time would produce the same result. In particular,
/// promotion of temporaries must not change program behavior; if the promoted could be
/// written to, that would be a problem.
pub struct HasMutInterior;

impl Qualif for HasMutInterior {
    const IDX: usize = 0;

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        !ty.is_freeze(cx.tcx, cx.param_env, DUMMY_SP)
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, per_local: &BitSet<Local>, rvalue: &Rvalue<'tcx>) -> bool {
        match *rvalue {
            // Returning `true` for `Rvalue::Ref` indicates the borrow isn't
            // allowed in constants (and the `Checker` will error), and/or it
            // won't be promoted, due to `&mut ...` or interior mutability.
            Rvalue::Ref(_, kind, ref place) => {
                let ty = place.ty(cx.body, cx.tcx).ty;

                if let BorrowKind::Mut { .. } = kind {
                    // In theory, any zero-sized value could be borrowed
                    // mutably without consequences.
                    match ty.kind {
                        // Inside a `static mut`, &mut [...] is also allowed.
                        ty::Array(..) | ty::Slice(_) if cx.mode == Mode::StaticMut => {},

                        // FIXME(ecstaticmorse): uncomment the following match arm to stop marking
                        // `&mut []` as `HasMutInterior`.
                        /*
                        ty::Array(_, len) if len.try_eval_usize(cx.tcx, cx.param_env) == Some(0)
                            => {},
                        */

                        _ => return true,
                    }
                }
            }

            Rvalue::Aggregate(ref kind, _) => {
                if let AggregateKind::Adt(def, ..) = **kind {
                    if Some(def.did) == cx.tcx.lang_items().unsafe_cell_type() {
                        let ty = rvalue.ty(cx.body, cx.tcx);
                        assert_eq!(Self::in_any_value_of_ty(cx, ty), true);
                        return true;
                    }
                }
            }

            _ => {}
        }

        Self::in_rvalue_structurally(cx, per_local, rvalue)
    }
}

/// Constant containing an ADT that implements `Drop`.
/// This must be ruled out (a) because we cannot run `Drop` during compile-time
/// as that might not be a `const fn`, and (b) because implicit promotion would
/// remove side-effects that occur as part of dropping that value.
pub struct NeedsDrop;

impl Qualif for NeedsDrop {
    const IDX: usize = 1;
    const IS_CLEARED_ON_MOVE: bool = true;

    fn in_any_value_of_ty(cx: &ConstCx<'_, 'tcx>, ty: Ty<'tcx>) -> bool {
        ty.needs_drop(cx.tcx, cx.param_env)
    }

    fn in_rvalue(cx: &ConstCx<'_, 'tcx>, per_local: &BitSet<Local>, rvalue: &Rvalue<'tcx>) -> bool {
        if let Rvalue::Aggregate(ref kind, _) = *rvalue {
            if let AggregateKind::Adt(def, ..) = **kind {
                if def.has_dtor(cx.tcx) {
                    return true;
                }
            }
        }

        Self::in_rvalue_structurally(cx, per_local, rvalue)
    }
}
