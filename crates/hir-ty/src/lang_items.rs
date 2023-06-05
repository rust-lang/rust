//! Functions to detect special lang items

use hir_def::{data::adt::StructFlags, lang_item::LangItem, AdtId};
use hir_expand::name::Name;

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
    use hir_expand::name;
    use syntax::ast::{ArithOp, BinaryOp, CmpOp, Ordering};
    Some(match op {
        BinaryOp::LogicOp(_) => return None,
        BinaryOp::ArithOp(aop) => match aop {
            ArithOp::Add => (name![add], LangItem::Add),
            ArithOp::Mul => (name![mul], LangItem::Mul),
            ArithOp::Sub => (name![sub], LangItem::Sub),
            ArithOp::Div => (name![div], LangItem::Div),
            ArithOp::Rem => (name![rem], LangItem::Rem),
            ArithOp::Shl => (name![shl], LangItem::Shl),
            ArithOp::Shr => (name![shr], LangItem::Shr),
            ArithOp::BitXor => (name![bitxor], LangItem::BitXor),
            ArithOp::BitOr => (name![bitor], LangItem::BitOr),
            ArithOp::BitAnd => (name![bitand], LangItem::BitAnd),
        },
        BinaryOp::Assignment { op: Some(aop) } => match aop {
            ArithOp::Add => (name![add_assign], LangItem::AddAssign),
            ArithOp::Mul => (name![mul_assign], LangItem::MulAssign),
            ArithOp::Sub => (name![sub_assign], LangItem::SubAssign),
            ArithOp::Div => (name![div_assign], LangItem::DivAssign),
            ArithOp::Rem => (name![rem_assign], LangItem::RemAssign),
            ArithOp::Shl => (name![shl_assign], LangItem::ShlAssign),
            ArithOp::Shr => (name![shr_assign], LangItem::ShrAssign),
            ArithOp::BitXor => (name![bitxor_assign], LangItem::BitXorAssign),
            ArithOp::BitOr => (name![bitor_assign], LangItem::BitOrAssign),
            ArithOp::BitAnd => (name![bitand_assign], LangItem::BitAndAssign),
        },
        BinaryOp::CmpOp(cop) => match cop {
            CmpOp::Eq { negated: false } => (name![eq], LangItem::PartialEq),
            CmpOp::Eq { negated: true } => (name![ne], LangItem::PartialEq),
            CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                (name![le], LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                (name![lt], LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                (name![ge], LangItem::PartialOrd)
            }
            CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                (name![gt], LangItem::PartialOrd)
            }
        },
        BinaryOp::Assignment { op: None } => return None,
    })
}
