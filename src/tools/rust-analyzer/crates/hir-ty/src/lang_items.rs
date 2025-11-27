//! Functions to detect special lang items

use hir_def::{AdtId, lang_item::LangItem, signatures::StructFlags};
use intern::{Symbol, sym};

use crate::db::HirDatabase;

pub fn is_box(db: &dyn HirDatabase, adt: AdtId) -> bool {
    let AdtId::StructId(id) = adt else { return false };
    db.struct_signature(id).flags.contains(StructFlags::IS_BOX)
}

pub fn lang_items_for_bin_op(op: syntax::ast::BinaryOp) -> Option<(Symbol, LangItem)> {
    use syntax::ast::{ArithOp, BinaryOp, CmpOp, Ordering};
    Some(match op {
        BinaryOp::LogicOp(_) => return None,
        BinaryOp::ArithOp(aop) => match aop {
            ArithOp::Add => (sym::add, LangItem::Add),
            ArithOp::Mul => (sym::mul, LangItem::Mul),
            ArithOp::Sub => (sym::sub, LangItem::Sub),
            ArithOp::Div => (sym::div, LangItem::Div),
            ArithOp::Rem => (sym::rem, LangItem::Rem),
            ArithOp::Shl => (sym::shl, LangItem::Shl),
            ArithOp::Shr => (sym::shr, LangItem::Shr),
            ArithOp::BitXor => (sym::bitxor, LangItem::BitXor),
            ArithOp::BitOr => (sym::bitor, LangItem::BitOr),
            ArithOp::BitAnd => (sym::bitand, LangItem::BitAnd),
        },
        BinaryOp::Assignment { op: Some(aop) } => match aop {
            ArithOp::Add => (sym::add_assign, LangItem::AddAssign),
            ArithOp::Mul => (sym::mul_assign, LangItem::MulAssign),
            ArithOp::Sub => (sym::sub_assign, LangItem::SubAssign),
            ArithOp::Div => (sym::div_assign, LangItem::DivAssign),
            ArithOp::Rem => (sym::rem_assign, LangItem::RemAssign),
            ArithOp::Shl => (sym::shl_assign, LangItem::ShlAssign),
            ArithOp::Shr => (sym::shr_assign, LangItem::ShrAssign),
            ArithOp::BitXor => (sym::bitxor_assign, LangItem::BitXorAssign),
            ArithOp::BitOr => (sym::bitor_assign, LangItem::BitOrAssign),
            ArithOp::BitAnd => (sym::bitand_assign, LangItem::BitAndAssign),
        },
        BinaryOp::CmpOp(cop) => match cop {
            CmpOp::Eq { negated: false } => (sym::eq, LangItem::PartialEq),
            CmpOp::Eq { negated: true } => (sym::ne, LangItem::PartialEq),
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                (sym::le, LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                (sym::lt, LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                (sym::ge, LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                (sym::gt, LangItem::PartialOrd)
            }
        },
        BinaryOp::Assignment { op: None } => return None,
    })
}
