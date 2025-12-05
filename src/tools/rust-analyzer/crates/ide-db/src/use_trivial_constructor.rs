//! Functionality for generating trivial constructors

use hir::StructKind;
use span::Edition;
use syntax::{
    ToSmolStr,
    ast::{Expr, Path, make},
};

/// given a type return the trivial constructor (if one exists)
pub fn use_trivial_constructor(
    db: &crate::RootDatabase,
    path: Path,
    ty: &hir::Type<'_>,
    edition: Edition,
) -> Option<Expr> {
    match ty.as_adt() {
        Some(hir::Adt::Enum(x)) => {
            if let &[variant] = &*x.variants(db)
                && variant.kind(db) == hir::StructKind::Unit
            {
                let path = make::path_qualified(
                    path,
                    make::path_segment(make::name_ref(
                        &variant.name(db).display_no_db(edition).to_smolstr(),
                    )),
                );

                return Some(make::expr_path(path));
            }
        }
        Some(hir::Adt::Struct(x)) if x.kind(db) == StructKind::Unit => {
            return Some(make::expr_path(path));
        }
        _ => {}
    }

    None
}
