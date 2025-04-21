//! Functions to detect special lang items

use hir_def::{AdtId, lang_item::LangItem, signatures::StructFlags};
use hir_expand::name::Name;
use intern::sym;

use crate::db::HirDatabase;

pub fn is_box(db: &dyn HirDatabase, adt: AdtId) -> bool {
    let AdtId::StructId(id) = adt else { return false };
    db.struct_signature(id).flags.contains(StructFlags::IS_BOX)
}

pub fn lang_items_for_bin_op(op: syntax::ast::BinaryOp) -> Option<(Name, LangItem)> {
    use syntax::ast::{ArithOp, BinaryOp, CmpOp, Ordering};
    Some(match op {
        BinaryOp::LogicOp(_) => return None,
        BinaryOp::ArithOp(aop) => match aop {
            ArithOp::Add => (Name::new_symbol_root(sym::add), LangItem::Add),
            ArithOp::Mul => (Name::new_symbol_root(sym::mul), LangItem::Mul),
            ArithOp::Sub => (Name::new_symbol_root(sym::sub), LangItem::Sub),
            ArithOp::Div => (Name::new_symbol_root(sym::div), LangItem::Div),
            ArithOp::Rem => (Name::new_symbol_root(sym::rem), LangItem::Rem),
            ArithOp::Shl => (Name::new_symbol_root(sym::shl), LangItem::Shl),
            ArithOp::Shr => (Name::new_symbol_root(sym::shr), LangItem::Shr),
            ArithOp::BitXor => (Name::new_symbol_root(sym::bitxor), LangItem::BitXor),
            ArithOp::BitOr => (Name::new_symbol_root(sym::bitor), LangItem::BitOr),
            ArithOp::BitAnd => (Name::new_symbol_root(sym::bitand), LangItem::BitAnd),
        },
        BinaryOp::Assignment { op: Some(aop) } => match aop {
            ArithOp::Add => (Name::new_symbol_root(sym::add_assign), LangItem::AddAssign),
            ArithOp::Mul => (Name::new_symbol_root(sym::mul_assign), LangItem::MulAssign),
            ArithOp::Sub => (Name::new_symbol_root(sym::sub_assign), LangItem::SubAssign),
            ArithOp::Div => (Name::new_symbol_root(sym::div_assign), LangItem::DivAssign),
            ArithOp::Rem => (Name::new_symbol_root(sym::rem_assign), LangItem::RemAssign),
            ArithOp::Shl => (Name::new_symbol_root(sym::shl_assign), LangItem::ShlAssign),
            ArithOp::Shr => (Name::new_symbol_root(sym::shr_assign), LangItem::ShrAssign),
            ArithOp::BitXor => (Name::new_symbol_root(sym::bitxor_assign), LangItem::BitXorAssign),
            ArithOp::BitOr => (Name::new_symbol_root(sym::bitor_assign), LangItem::BitOrAssign),
            ArithOp::BitAnd => (Name::new_symbol_root(sym::bitand_assign), LangItem::BitAndAssign),
        },
        BinaryOp::CmpOp(cop) => match cop {
            CmpOp::Eq { negated: false } => (Name::new_symbol_root(sym::eq), LangItem::PartialEq),
            CmpOp::Eq { negated: true } => (Name::new_symbol_root(sym::ne), LangItem::PartialEq),
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                (Name::new_symbol_root(sym::le), LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                (Name::new_symbol_root(sym::lt), LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                (Name::new_symbol_root(sym::ge), LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                (Name::new_symbol_root(sym::gt), LangItem::PartialOrd)
            }
        },
        BinaryOp::Assignment { op: None } => return None,
    })
}
