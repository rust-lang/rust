//! Functions to detect special lang items

use hir_def::{lang_item::LangItem, AdtId, HasModule};

use crate::db::HirDatabase;

pub fn is_box(adt: AdtId, db: &dyn HirDatabase) -> bool {
    let krate = adt.module(db.upcast()).krate();
    let box_adt =
        db.lang_item(krate, LangItem::OwnedBox).and_then(|it| it.as_struct()).map(AdtId::from);
    Some(adt) == box_adt
}

pub fn is_unsafe_cell(adt: AdtId, db: &dyn HirDatabase) -> bool {
    let krate = adt.module(db.upcast()).krate();
    let box_adt =
        db.lang_item(krate, LangItem::UnsafeCell).and_then(|it| it.as_struct()).map(AdtId::from);
    Some(adt) == box_adt
}
