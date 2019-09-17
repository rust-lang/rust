use hir::Either;
use ra_syntax::{ast, AstNode, AstPtr};
use test_utils::tested_by;

use crate::db::RootDatabase;

pub enum NameRefKind {
    Method(hir::Function),
    Macro(hir::MacroDef),
    FieldAccess(hir::StructField),
    AssocItem(hir::AssocItem),
    Def(hir::ModuleDef),
    SelfType(hir::Ty),
    Pat(AstPtr<ast::BindPat>),
    SelfParam(AstPtr<ast::SelfParam>),
    GenericParam(u32),
}

pub(crate) fn classify_name_ref(
    db: &RootDatabase,
    analyzer: &hir::SourceAnalyzer,
    name_ref: &ast::NameRef,
) -> Option<NameRefKind> {
    use NameRefKind::*;

    // Check if it is a method
    if let Some(method_call) = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast) {
        tested_by!(goto_definition_works_for_methods);
        if let Some(func) = analyzer.resolve_method_call(&method_call) {
            return Some(Method(func));
        }
    }

    // It could be a macro call
    if let Some(macro_call) = name_ref
        .syntax()
        .parent()
        .and_then(|node| node.parent())
        .and_then(|node| node.parent())
        .and_then(ast::MacroCall::cast)
    {
        tested_by!(goto_definition_works_for_macros);
        if let Some(mac) = analyzer.resolve_macro_call(db, &macro_call) {
            return Some(Macro(mac));
        }
    }

    // It could also be a field access
    if let Some(field_expr) = name_ref.syntax().parent().and_then(ast::FieldExpr::cast) {
        tested_by!(goto_definition_works_for_fields);
        if let Some(field) = analyzer.resolve_field(&field_expr) {
            return Some(FieldAccess(field));
        };
    }

    // It could also be a named field
    if let Some(field_expr) = name_ref.syntax().parent().and_then(ast::RecordField::cast) {
        tested_by!(goto_definition_works_for_record_fields);

        let record_lit = field_expr.syntax().ancestors().find_map(ast::RecordLit::cast);

        if let Some(ty) = record_lit.and_then(|lit| analyzer.type_of(db, &lit.into())) {
            if let Some((hir::Adt::Struct(s), _)) = ty.as_adt() {
                let hir_path = hir::Path::from_name_ref(name_ref);
                let hir_name = hir_path.as_ident().unwrap();

                if let Some(field) = s.field(db, hir_name) {
                    return Some(FieldAccess(field));
                }
            }
        }
    }

    // General case, a path or a local:
    if let Some(path) = name_ref.syntax().ancestors().find_map(ast::Path::cast) {
        if let Some(resolved) = analyzer.resolve_path(db, &path) {
            return match resolved {
                hir::PathResolution::Def(def) => Some(Def(def)),
                hir::PathResolution::LocalBinding(Either::A(pat)) => Some(Pat(pat)),
                hir::PathResolution::LocalBinding(Either::B(par)) => Some(SelfParam(par)),
                hir::PathResolution::GenericParam(par) => {
                    // FIXME: get generic param def
                    Some(GenericParam(par))
                }
                hir::PathResolution::Macro(def) => Some(Macro(def)),
                hir::PathResolution::SelfType(impl_block) => {
                    let ty = impl_block.target_ty(db);
                    Some(SelfType(ty))
                }
                hir::PathResolution::AssocItem(assoc) => Some(AssocItem(assoc)),
            };
        }
    }

    None
}
