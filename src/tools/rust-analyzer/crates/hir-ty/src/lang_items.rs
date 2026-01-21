//! Functions to detect special lang items

use hir_def::{AdtId, TraitId, lang_item::LangItems, signatures::StructFlags};
use intern::{Symbol, sym};

use crate::db::HirDatabase;

pub fn is_box(db: &dyn HirDatabase, adt: AdtId) -> bool {
    let AdtId::StructId(id) = adt else { return false };
    db.struct_signature(id).flags.contains(StructFlags::IS_BOX)
}

pub fn lang_items_for_bin_op(
    lang_items: &LangItems,
    op: syntax::ast::BinaryOp,
) -> Option<(Symbol, Option<TraitId>)> {
    use syntax::ast::{ArithOp, BinaryOp, CmpOp, Ordering};
    Some(match op {
        BinaryOp::LogicOp(_) => return None,
        BinaryOp::ArithOp(aop) => match aop {
            ArithOp::Add => (sym::add, lang_items.Add),
            ArithOp::Mul => (sym::mul, lang_items.Mul),
            ArithOp::Sub => (sym::sub, lang_items.Sub),
            ArithOp::Div => (sym::div, lang_items.Div),
            ArithOp::Rem => (sym::rem, lang_items.Rem),
            ArithOp::Shl => (sym::shl, lang_items.Shl),
            ArithOp::Shr => (sym::shr, lang_items.Shr),
            ArithOp::BitXor => (sym::bitxor, lang_items.BitXor),
            ArithOp::BitOr => (sym::bitor, lang_items.BitOr),
            ArithOp::BitAnd => (sym::bitand, lang_items.BitAnd),
        },
        BinaryOp::Assignment { op: Some(aop) } => match aop {
            ArithOp::Add => (sym::add_assign, lang_items.AddAssign),
            ArithOp::Mul => (sym::mul_assign, lang_items.MulAssign),
            ArithOp::Sub => (sym::sub_assign, lang_items.SubAssign),
            ArithOp::Div => (sym::div_assign, lang_items.DivAssign),
            ArithOp::Rem => (sym::rem_assign, lang_items.RemAssign),
            ArithOp::Shl => (sym::shl_assign, lang_items.ShlAssign),
            ArithOp::Shr => (sym::shr_assign, lang_items.ShrAssign),
            ArithOp::BitXor => (sym::bitxor_assign, lang_items.BitXorAssign),
            ArithOp::BitOr => (sym::bitor_assign, lang_items.BitOrAssign),
            ArithOp::BitAnd => (sym::bitand_assign, lang_items.BitAndAssign),
        },
        BinaryOp::CmpOp(cop) => match cop {
            CmpOp::Eq { negated: false } => (sym::eq, lang_items.PartialEq),
            CmpOp::Eq { negated: true } => (sym::ne, lang_items.PartialEq),
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                (sym::le, lang_items.PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                (sym::lt, lang_items.PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                (sym::ge, lang_items.PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                (sym::gt, lang_items.PartialOrd)
            }
        },
        BinaryOp::Assignment { op: None } => return None,
    })
}
