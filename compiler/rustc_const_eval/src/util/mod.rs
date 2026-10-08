use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::mir;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

mod alignment;
pub(crate) mod caller_location;
mod check_validity_requirement;
mod compare_types;
mod type_name;

pub use self::alignment::{most_packed_projection, place_unalignment};
pub use self::check_validity_requirement::check_validity_requirement;
pub(crate) use self::check_validity_requirement::validate_scalar_in_layout;
pub use self::compare_types::{relate_types, sub_types};
pub use self::type_name::type_name;

/// Classify whether an operator is "left-homogeneous", i.e., the LHS has the
/// same type as the result.
#[inline]
pub fn binop_left_homogeneous(op: mir::BinOp) -> bool {
    use mir::BinOp::*;
    match op {
        Add | AddUnchecked | Sub | SubUnchecked | Mul | MulUnchecked | Div | Rem | BitXor
        | BitAnd | BitOr | Offset | Shl | ShlUnchecked | Shr | ShrUnchecked => true,
        AddWithOverflow | SubWithOverflow | MulWithOverflow | Eq | Ne | Lt | Le | Gt | Ge | Cmp => {
            false
        }
    }
}

/// Classify whether an operator is "right-homogeneous", i.e., the RHS has the
/// same type as the LHS.
#[inline]
pub fn binop_right_homogeneous(op: mir::BinOp) -> bool {
    use mir::BinOp::*;
    match op {
        Add | AddUnchecked | AddWithOverflow | Sub | SubUnchecked | SubWithOverflow | Mul
        | MulUnchecked | MulWithOverflow | Div | Rem | BitXor | BitAnd | BitOr | Eq | Ne | Lt
        | Le | Gt | Ge | Cmp => true,
        Offset | Shl | ShlUnchecked | Shr | ShrUnchecked => false,
    }
}

/// Classify whether an operator is "homogeneous", i.e., the operand has the
/// same type as the result.
#[inline]
pub fn unop_homogeneous(op: mir::UnOp) -> bool {
    match op {
        mir::UnOp::Not | mir::UnOp::Neg => true,
        mir::UnOp::PtrMetadata => false,
    }
}

pub fn context_spans(tcx: TyCtxt<'_>, span: Span, def_id: DefId) -> Vec<Span> {
    // For consts where the enclosing item might be useful context, include their spans in the
    // diagnostic.
    let instance_span = tcx.def_span(def_id).shrink_to_lo();
    let parent = tcx.parent(def_id);
    let parent_span = tcx.def_span(parent).shrink_to_lo();
    let mut span = span;
    if let Some(sp) = span.macro_backtrace().last() {
        // We want the outermost span when macros are involved, as we don't care about the
        // macro's internal consts for the purposes of adding context.
        span = sp.call_site;
    }
    match (tcx.def_kind(def_id), tcx.def_kind(parent)) {
        (
            _,
            DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Trait
            | DefKind::Impl { .. }
            | DefKind::TyAlias
            | DefKind::Const
            | DefKind::Fn
            | DefKind::Static { .. },
        )
        | (DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy, _)
            if span.eq_ctxt(instance_span) && span.eq_ctxt(parent_span) =>
        {
            vec![instance_span, parent_span]
        }
        (_, DefKind::Variant) if span.eq_ctxt(instance_span) && span.eq_ctxt(parent_span) => {
            vec![instance_span, parent_span, tcx.def_span(tcx.parent(parent)).shrink_to_lo()]
        }
        (DefKind::Const | DefKind::Static { .. }, _) if span.eq_ctxt(instance_span) => {
            vec![instance_span]
        }
        _ => vec![],
    }
}
