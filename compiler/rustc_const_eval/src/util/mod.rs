use rustc_middle::mir;

mod alignment;
pub(crate) mod caller_location;
mod check_validity_requirement;
mod compare_types;
mod type_name;

pub use self::alignment::{is_disaligned, is_within_packed};
pub use self::check_validity_requirement::check_validity_requirement;
pub(crate) use self::check_validity_requirement::validate_scalar_in_layout;
pub use self::compare_types::{relate_types, sub_types};
pub use self::type_name::type_name;

/// Classify whether an operator is "left-homogeneous", i.e., the LHS has the
/// same type as the result.
#[inline]
pub fn binop_left_homogeneous(op: mir::BinOp) -> bool {
    use rustc_middle::mir::BinOp::*;
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
    use rustc_middle::mir::BinOp::*;
    match op {
        Add | AddUnchecked | AddWithOverflow | Sub | SubUnchecked | SubWithOverflow | Mul
        | MulUnchecked | MulWithOverflow | Div | Rem | BitXor | BitAnd | BitOr | Eq | Ne | Lt
        | Le | Gt | Ge | Cmp => true,
        Offset | Shl | ShlUnchecked | Shr | ShrUnchecked => false,
    }
}
