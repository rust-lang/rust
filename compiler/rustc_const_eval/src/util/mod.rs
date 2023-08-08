use rustc_middle::mir;

mod alignment;
mod check_validity_requirement;
mod compare_types;
mod type_name;

pub use self::alignment::is_disaligned;
pub use self::check_validity_requirement::check_validity_requirement;
pub use self::compare_types::{is_equal_up_to_subtyping, is_subtype};
pub use self::type_name::type_name;

/// Classify whether an operator is "left-homogeneous", i.e., the LHS has the
/// same type as the result.
#[inline]
pub(crate) fn binop_left_homogeneous(op: mir::BinOp) -> bool {
    use rustc_middle::mir::BinOp::*;
    match op {
        Add | AddUnchecked | Sub | SubUnchecked | Mul | MulUnchecked | Div | Rem | BitXor
        | BitAnd | BitOr | Offset | Shl | ShlUnchecked | Shr | ShrUnchecked => true,
        Eq | Ne | Lt | Le | Gt | Ge => false,
    }
}

/// Classify whether an operator is "right-homogeneous", i.e., the RHS has the
/// same type as the LHS.
#[inline]
pub(crate) fn binop_right_homogeneous(op: mir::BinOp) -> bool {
    use rustc_middle::mir::BinOp::*;
    match op {
        Add | AddUnchecked | Sub | SubUnchecked | Mul | MulUnchecked | Div | Rem | BitXor
        | BitAnd | BitOr | Eq | Ne | Lt | Le | Gt | Ge => true,
        Offset | Shl | ShlUnchecked | Shr | ShrUnchecked => false,
    }
}
