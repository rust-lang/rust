//! Functions to detect special lang items

use hir_def::{AdtId, HasModule};
use hir_expand::name;

use crate::db::HirDatabase;

pub fn is_box(adt: AdtId, db: &dyn HirDatabase) -> bool {
    let owned_box = name![owned_box].to_smol_str();
    let krate = adt.module(db.upcast()).krate();
    let box_adt = db.lang_item(krate, owned_box).and_then(|it| it.as_struct()).map(AdtId::from);
    Some(adt) == box_adt
}

pub fn is_unsafe_cell(adt: AdtId, db: &dyn HirDatabase) -> bool {
    let owned_box = name![unsafe_cell].to_smol_str();
    let krate = adt.module(db.upcast()).krate();
    let box_adt = db.lang_item(krate, owned_box).and_then(|it| it.as_struct()).map(AdtId::from);
    Some(adt) == box_adt
}
