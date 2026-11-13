//! Functions to detect special lang items

use hir_def::{FunctionId, TraitId, lang_item::LangItems};

pub fn lang_items_for_bin_op(
    lang_items: &LangItems,
    op: syntax::ast::BinaryOp,
) -> Option<(Option<FunctionId>, Option<TraitId>)> {
    use syntax::ast::{ArithOp, BinaryOp, CmpOp, Ordering};
    Some(match op {
        BinaryOp::LogicOp(_) => return None,
        BinaryOp::ArithOp(aop) => match aop {
            ArithOp::Add => (lang_items.Add_add, lang_items.Add),
            ArithOp::Mul => (lang_items.Mul_mul, lang_items.Mul),
            ArithOp::Sub => (lang_items.Sub_sub, lang_items.Sub),
            ArithOp::Div => (lang_items.Div_div, lang_items.Div),
            ArithOp::Rem => (lang_items.Rem_rem, lang_items.Rem),
            ArithOp::Shl => (lang_items.Shl_shl, lang_items.Shl),
            ArithOp::Shr => (lang_items.Shr_shr, lang_items.Shr),
            ArithOp::BitXor => (lang_items.BitXor_bitxor, lang_items.BitXor),
            ArithOp::BitOr => (lang_items.BitOr_bitor, lang_items.BitOr),
            ArithOp::BitAnd => (lang_items.BitAnd_bitand, lang_items.BitAnd),
        },
        BinaryOp::Assignment { op: Some(aop) } => match aop {
            ArithOp::Add => (lang_items.AddAssign_add_assign, lang_items.AddAssign),
            ArithOp::Mul => (lang_items.MulAssign_mul_assign, lang_items.MulAssign),
            ArithOp::Sub => (lang_items.SubAssign_sub_assign, lang_items.SubAssign),
            ArithOp::Div => (lang_items.DivAssign_div_assign, lang_items.DivAssign),
            ArithOp::Rem => (lang_items.RemAssign_rem_assign, lang_items.RemAssign),
            ArithOp::Shl => (lang_items.ShlAssign_shl_assign, lang_items.ShlAssign),
            ArithOp::Shr => (lang_items.ShrAssign_shr_assign, lang_items.ShrAssign),
            ArithOp::BitXor => (lang_items.BitXorAssign_bitxor_assign, lang_items.BitXorAssign),
            ArithOp::BitOr => (lang_items.BitOrAssign_bitor_assign, lang_items.BitOrAssign),
            ArithOp::BitAnd => (lang_items.BitAndAssign_bitand_assign, lang_items.BitAndAssign),
        },
        BinaryOp::CmpOp(cop) => match cop {
            CmpOp::Eq { negated: false } => (lang_items.PartialEq_eq, lang_items.PartialEq),
            CmpOp::Eq { negated: true } => (lang_items.PartialEq_ne, lang_items.PartialEq),
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                (lang_items.PartialOrd_le, lang_items.PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                (lang_items.PartialOrd_lt, lang_items.PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                (lang_items.PartialOrd_ge, lang_items.PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                (lang_items.PartialOrd_gt, lang_items.PartialOrd)
            }
        },
        BinaryOp::Assignment { op: None } => return None,
    })
}
