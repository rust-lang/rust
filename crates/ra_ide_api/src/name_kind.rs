//! FIXME: write short doc here

use hir::{Either, FromSource};
use ra_db::FileId;
use ra_syntax::{ast, AstNode, AstPtr};
use test_utils::tested_by;

use crate::db::RootDatabase;

pub enum NameKind {
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
) -> Option<NameKind> {
    use NameKind::*;

    // Check if it is a method
    if let Some(method_call) = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast) {
        tested_by!(goto_definition_works_for_methods);
        if let Some(func) = analyzer.resolve_method_call(&method_call) {
            return Some(AssocItem(func.into()));
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

        if let Some(record_lit) = field_expr.syntax().ancestors().find_map(ast::RecordLit::cast) {
            let variant_def = analyzer.resolve_record_literal(&record_lit)?;
            let hir_path = hir::Path::from_name_ref(name_ref);
            let hir_name = hir_path.as_ident()?;
            let field = variant_def.field(db, hir_name)?;
            return Some(FieldAccess(field));
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

pub(crate) fn classify_name(
    db: &RootDatabase,
    file_id: FileId,
    name: &ast::Name,
) -> Option<NameKind> {
    use NameKind::*;

    let parent = name.syntax().parent()?;
    let file_id = file_id.into();

    if let Some(pat) = ast::BindPat::cast(parent.clone()) {
        return Some(Pat(AstPtr::new(&pat)));
    }
    if let Some(var) = ast::EnumVariant::cast(parent.clone()) {
        let src = hir::Source { file_id, ast: var };
        let var = hir::EnumVariant::from_source(db, src)?;
        return Some(Def(var.into()));
    }
    if let Some(field) = ast::RecordFieldDef::cast(parent.clone()) {
        let src = hir::Source { file_id, ast: hir::FieldSource::Named(field) };
        let field = hir::StructField::from_source(db, src)?;
        return Some(FieldAccess(field));
    }
    if let Some(field) = ast::TupleFieldDef::cast(parent.clone()) {
        let src = hir::Source { file_id, ast: hir::FieldSource::Pos(field) };
        let field = hir::StructField::from_source(db, src)?;
        return Some(FieldAccess(field));
    }
    if let Some(_) = parent.parent().and_then(ast::ItemList::cast) {
        let ast = ast::ImplItem::cast(parent.clone())?;
        let src = hir::Source { file_id, ast };
        let item = hir::AssocItem::from_source(db, src)?;
        return Some(AssocItem(item));
    }
    if let Some(item) = ast::ModuleItem::cast(parent.clone()) {
        let src = hir::Source { file_id, ast: item };
        let def = hir::ModuleDef::from_source(db, src)?;
        return Some(Def(def));
    }
    // FIXME: TYPE_PARAM, ALIAS, MACRO_CALL; Union

    None
}
