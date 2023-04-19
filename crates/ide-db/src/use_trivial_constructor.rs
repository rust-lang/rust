//! Functionality for generating trivial constructors

use hir::StructKind;
use syntax::ast;

/// given a type return the trivial constructor (if one exists)
pub fn use_trivial_constructor(
    db: &crate::RootDatabase,
    path: ast::Path,
    ty: &hir::Type,
) -> Option<ast::Expr> {
    match ty.as_adt() {
        Some(hir::Adt::Enum(x)) => {
            if let &[variant] = &*x.variants(db) {
                if variant.kind(db) == hir::StructKind::Unit {
                    let path = ast::make::path_qualified(
                        path,
                        syntax::ast::make::path_segment(ast::make::name_ref(
                            &variant.name(db).to_smol_str(),
                        )),
                    );

                    return Some(syntax::ast::make::expr_path(path));
                }
            }
        }
        Some(hir::Adt::Struct(x)) if x.kind(db) == StructKind::Unit => {
            return Some(syntax::ast::make::expr_path(path));
        }
        _ => {}
    }

    None
}
