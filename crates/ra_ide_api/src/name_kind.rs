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

    macro_rules! match_ast {
        (match $node:ident {
            $( ast::$ast:ident($it:ident) => $res:block, )*
            _ => $catch_all:expr,
        }) => {{
            $( if let Some($it) = ast::$ast::cast($node.clone()) $res else )*
            { $catch_all }
        }};
    }

    // FIXME: add ast::MacroCall(it)
    match_ast! {
        match parent {
            ast::BindPat(it) => {
                let pat = AstPtr::new(&it);
                Some(Pat(pat))
            },
            ast::RecordFieldDef(it) => {
                let src = hir::Source { file_id, ast: hir::FieldSource::Named(it) };
                let field = hir::StructField::from_source(db, src)?;
                Some(FieldAccess(field))
            },
            ast::FnDef(it) => {
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    let src = hir::Source { file_id, ast: ast::ImplItem::from(it) };
                    let item = hir::AssocItem::from_source(db, src)?;
                    Some(AssocItem(item))
                } else {
                    let src = hir::Source { file_id, ast: it };
                    let def = hir::Function::from_source(db, src)?;
                    Some(Def(def.into()))
                }
            },
            ast::ConstDef(it) => {
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    let src = hir::Source { file_id, ast: ast::ImplItem::from(it) };
                    let item = hir::AssocItem::from_source(db, src)?;
                    Some(AssocItem(item))
                } else {
                    let src = hir::Source { file_id, ast: it };
                    let def = hir::Const::from_source(db, src)?;
                    Some(Def(def.into()))
                }
            },
            ast::TypeAliasDef(it) => {
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    let src = hir::Source { file_id, ast: ast::ImplItem::from(it) };
                    let item = hir::AssocItem::from_source(db, src)?;
                    Some(AssocItem(item))
                } else {
                    let src = hir::Source { file_id, ast: it };
                    let def = hir::TypeAlias::from_source(db, src)?;
                    Some(Def(def.into()))
                }
            },
            ast::Module(it) => {
                let src = hir::Source { file_id, ast: hir::ModuleSource::Module(it) };
                let def = hir::Module::from_definition(db, src)?;
                Some(Def(def.into()))
            },
            ast::StructDef(it) => {
                let src = hir::Source { file_id, ast: it };
                let def = hir::Struct::from_source(db, src)?;
                Some(Def(def.into()))
            },
            ast::EnumDef(it) => {
                let src = hir::Source { file_id, ast: it };
                let def = hir::Enum::from_source(db, src)?;
                Some(Def(def.into()))
            },
            ast::TraitDef(it) => {
                let src = hir::Source { file_id, ast: it };
                let def = hir::Trait::from_source(db, src)?;
                Some(Def(def.into()))
            },
            ast::StaticDef(it) => {
                let src = hir::Source { file_id, ast: it };
                let def = hir::Static::from_source(db, src)?;
                Some(Def(def.into()))
            },
            ast::EnumVariant(it) => {
                let src = hir::Source { file_id, ast: it };
                let def = hir::EnumVariant::from_source(db, src)?;
                Some(Def(def.into()))
            },
            _ => None,
        }
    }
}
