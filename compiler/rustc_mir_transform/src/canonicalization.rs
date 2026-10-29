use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};

/// Canonicalize operands of binary operations.
/// This only canonicalize their ordering for now.
pub(super) fn canonicalize_binary_operands<'tcx, 'body>(
    tcx: TyCtxt<'tcx>,
    local_decls: &'body LocalDecls<'tcx>,
    op: BinOp,
    lhs: &mut Operand<'tcx>,
    rhs: &mut Operand<'tcx>,
) -> bool {
    fn needs_swap<T: std::cmp::Ord>(lhs: &T, rhs: &T) -> Option<bool> {
        use std::cmp::Ordering;

        match lhs.cmp(rhs) {
            Ordering::Less => Some(false),
            Ordering::Equal => None,
            Ordering::Greater => Some(true),
        }
    }

    fn projection_rank<'tcx>(proj: &ProjectionElem<Local, Ty<'tcx>>) -> usize {
        match proj {
            ProjectionElem::Deref => 0,
            ProjectionElem::Field(..) => 1,
            ProjectionElem::Index(_) => 2,
            ProjectionElem::ConstantIndex { .. } => 3,
            ProjectionElem::Subslice { .. } => 4,
            ProjectionElem::Downcast(..) => 5,
            ProjectionElem::OpaqueCast(_) => 6,
            ProjectionElem::Transmute(_) => 7,
            ProjectionElem::PhantomDeref => 8,
        }
    }

    fn needs_swap_operands<'tcx>(lhs: &Operand<'tcx>, rhs: &Operand<'tcx>) -> bool {
        let (lhs, rhs) = match (lhs, rhs) {
            (Operand::Constant(_), Operand::Copy(_) | Operand::Move(_)) => return true,
            (Operand::Copy(lhs) | Operand::Move(lhs), Operand::Copy(rhs) | Operand::Move(rhs)) => {
                (lhs, rhs)
            }
            _ => return false,
        };

        if let Some(ns) = needs_swap(&lhs.local, &rhs.local) {
            return ns;
        }

        let lhs_proj = lhs.projection;
        let rhs_proj = rhs.projection;

        if let Some(ns) = needs_swap(&lhs_proj.len(), &rhs_proj.len()) {
            return ns;
        }

        for (lhs_proj, rhs_proj) in lhs_proj.iter().zip(rhs_proj.iter()) {
            if let Some(ns) = match (lhs_proj, rhs_proj) {
                (ProjectionElem::Field(lhs, _), ProjectionElem::Field(rhs, _)) => {
                    needs_swap(&lhs, &rhs)
                }

                (ProjectionElem::Index(lhs), ProjectionElem::Index(rhs)) => needs_swap(&lhs, &rhs),

                (
                    ProjectionElem::ConstantIndex { offset: l0, min_length: l1, from_end: l2 },
                    ProjectionElem::ConstantIndex { offset: r0, min_length: r1, from_end: r2 },
                ) => needs_swap(&(l0, l1, l2), &(r0, r1, r2)),

                (
                    ProjectionElem::Subslice { from: l0, to: l1, from_end: l2 },
                    ProjectionElem::Subslice { from: r0, to: r1, from_end: r2 },
                ) => needs_swap(&(l0, l1, l2), &(r0, r1, r2)),

                (ProjectionElem::Downcast(l0, l1), ProjectionElem::Downcast(r0, r1)) => {
                    needs_swap(&(l0, l1), &(r0, r1))
                }

                _ => needs_swap(&projection_rank(&lhs_proj), &projection_rank(&rhs_proj)),
            } {
                return ns;
            }
        }

        false
    }

    let needs_integral = match op {
        BinOp::Eq | BinOp::Ne => false,

        BinOp::Add
        | BinOp::AddUnchecked
        | BinOp::AddWithOverflow
        | BinOp::Mul
        | BinOp::MulUnchecked
        | BinOp::MulWithOverflow
        | BinOp::BitXor
        | BinOp::BitAnd
        | BinOp::BitOr => true,

        BinOp::Sub
        | BinOp::SubUnchecked
        | BinOp::SubWithOverflow
        | BinOp::Div
        | BinOp::Rem
        | BinOp::Shl
        | BinOp::ShlUnchecked
        | BinOp::Shr
        | BinOp::ShrUnchecked
        | BinOp::Lt
        | BinOp::Le
        | BinOp::Ge
        | BinOp::Gt
        | BinOp::Cmp
        | BinOp::Offset => return false,
    };

    if needs_integral && !lhs.ty(local_decls, tcx).is_integral() {
        return false;
    }

    if needs_swap_operands(lhs, rhs) {
        std::mem::swap(lhs, rhs);
        true
    } else {
        false
    }
}
