//! Functions to detect special lang items

use hir_def::{data::adt::StructFlags, lang_item::LangItem, AdtId};
use hir_expand::name::Name;
use intern::sym;

use crate::db::HirDatabase;

pub fn is_box(db: &dyn HirDatabase, adt: AdtId) -> bool {
    let AdtId::StructId(id) = adt else { return false };
    db.struct_data(id).flags.contains(StructFlags::IS_BOX)
}

pub fn is_unsafe_cell(db: &dyn HirDatabase, adt: AdtId) -> bool {
    let AdtId::StructId(id) = adt else { return false };

    db.struct_data(id).flags.contains(StructFlags::IS_UNSAFE_CELL)
}

pub fn lang_items_for_bin_op(op: syntax::ast::BinaryOp) -> Option<(Name, LangItem)> {
    use syntax::ast::{ArithOp, BinaryOp, CmpOp, Ordering};
    Some(match op {
        BinaryOp::LogicOp(_) => return None,
        BinaryOp::ArithOp(aop) => match aop {
            ArithOp::Add => (Name::new_symbol_root(sym::add.clone()), LangItem::Add),
            ArithOp::Mul => (Name::new_symbol_root(sym::mul.clone()), LangItem::Mul),
            ArithOp::Sub => (Name::new_symbol_root(sym::sub.clone()), LangItem::Sub),
            ArithOp::Div => (Name::new_symbol_root(sym::div.clone()), LangItem::Div),
            ArithOp::Rem => (Name::new_symbol_root(sym::rem.clone()), LangItem::Rem),
            ArithOp::Shl => (Name::new_symbol_root(sym::shl.clone()), LangItem::Shl),
            ArithOp::Shr => (Name::new_symbol_root(sym::shr.clone()), LangItem::Shr),
            ArithOp::BitXor => (Name::new_symbol_root(sym::bitxor.clone()), LangItem::BitXor),
            ArithOp::BitOr => (Name::new_symbol_root(sym::bitor.clone()), LangItem::BitOr),
            ArithOp::BitAnd => (Name::new_symbol_root(sym::bitand.clone()), LangItem::BitAnd),
        },
        BinaryOp::Assignment { op: Some(aop) } => match aop {
            ArithOp::Add => (Name::new_symbol_root(sym::add_assign.clone()), LangItem::AddAssign),
            ArithOp::Mul => (Name::new_symbol_root(sym::mul_assign.clone()), LangItem::MulAssign),
            ArithOp::Sub => (Name::new_symbol_root(sym::sub_assign.clone()), LangItem::SubAssign),
            ArithOp::Div => (Name::new_symbol_root(sym::div_assign.clone()), LangItem::DivAssign),
            ArithOp::Rem => (Name::new_symbol_root(sym::rem_assign.clone()), LangItem::RemAssign),
            ArithOp::Shl => (Name::new_symbol_root(sym::shl_assign.clone()), LangItem::ShlAssign),
            ArithOp::Shr => (Name::new_symbol_root(sym::shr_assign.clone()), LangItem::ShrAssign),
            ArithOp::BitXor => {
                (Name::new_symbol_root(sym::bitxor_assign.clone()), LangItem::BitXorAssign)
            }
            ArithOp::BitOr => {
                (Name::new_symbol_root(sym::bitor_assign.clone()), LangItem::BitOrAssign)
            }
            ArithOp::BitAnd => {
                (Name::new_symbol_root(sym::bitand_assign.clone()), LangItem::BitAndAssign)
            }
        },
        BinaryOp::CmpOp(cop) => match cop {
            CmpOp::Eq { negated: false } => {
                (Name::new_symbol_root(sym::eq.clone()), LangItem::PartialEq)
            }
            CmpOp::Eq { negated: true } => {
                (Name::new_symbol_root(sym::ne.clone()), LangItem::PartialEq)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                (Name::new_symbol_root(sym::le.clone()), LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                (Name::new_symbol_root(sym::lt.clone()), LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                (Name::new_symbol_root(sym::ge.clone()), LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                (Name::new_symbol_root(sym::gt.clone()), LangItem::PartialOrd)
            }
        },
        BinaryOp::Assignment { op: None } => return None,
    })
}
