use hir::{
    AssocItem, Either, EnumVariant, FromSource, Module, ModuleDef, ModuleSource, Path,
    PathResolution, Source, SourceAnalyzer, StructField,
};
use ra_db::FileId;
use ra_syntax::{ast, match_ast, AstNode, AstPtr};

use super::{definition::HasDefinition, Definition, NameKind};
use crate::db::RootDatabase;

use hir::{db::AstDatabase, HirFileId};

pub(crate) fn classify_name(
    db: &RootDatabase,
    file_id: FileId,
    name: &ast::Name,
) -> Option<Definition> {
    let parent = name.syntax().parent()?;
    let file_id = file_id.into();

    match_ast! {
        match parent {
            ast::BindPat(it) => {
                decl_from_pat(db, file_id, AstPtr::new(&it))
            },
            ast::RecordFieldDef(it) => {
                StructField::from_def(db, file_id, it)
            },
            ast::ImplItem(it) => {
                AssocItem::from_def(db, file_id, it.clone()).or_else(|| {
                    match it {
                        ast::ImplItem::FnDef(f) => ModuleDef::from_def(db, file_id, f.into()),
                        ast::ImplItem::ConstDef(c) => ModuleDef::from_def(db, file_id, c.into()),
                        ast::ImplItem::TypeAliasDef(a) => ModuleDef::from_def(db, file_id, a.into()),
                    }
                })
            },
            ast::EnumVariant(it) => {
                let src = hir::Source { file_id, ast: it.clone() };
                let def: ModuleDef = EnumVariant::from_source(db, src)?.into();
                Some(def.definition(db))
            },
            ast::ModuleItem(it) => {
                ModuleDef::from_def(db, file_id, it)
            },
            _ => None,
        }
    }
}

pub(crate) fn classify_name_ref(
    db: &RootDatabase,
    file_id: FileId,
    name_ref: &ast::NameRef,
) -> Option<Definition> {
    let analyzer = SourceAnalyzer::new(db, file_id, name_ref.syntax(), None);
    let parent = name_ref.syntax().parent()?;
    match_ast! {
        match parent {
            ast::MethodCallExpr(it) => {
                return AssocItem::from_ref(db, &analyzer, it);
            },
            ast::FieldExpr(it) => {
                if let Some(field) = analyzer.resolve_field(&it) {
                    return Some(field.definition(db));
                }
            },
            ast::RecordField(it) => {
                if let Some(record_lit) = it.syntax().ancestors().find_map(ast::RecordLit::cast) {
                    let variant_def = analyzer.resolve_record_literal(&record_lit)?;
                    let hir_path = Path::from_name_ref(name_ref);
                    let hir_name = hir_path.as_ident()?;
                    let field = variant_def.field(db, hir_name)?;
                    return Some(field.definition(db));
                }
            },
            _ => (),
        }
    }

    let ast = ModuleSource::from_child_node(db, file_id, &parent);
    let file_id = file_id.into();
    let container = Module::from_definition(db, Source { file_id, ast })?;
    let visibility = None;

    if let Some(macro_call) =
        parent.parent().and_then(|node| node.parent()).and_then(ast::MacroCall::cast)
    {
        if let Some(mac) = analyzer.resolve_macro_call(db, &macro_call) {
            return Some(Definition { item: NameKind::Macro(mac), container, visibility });
        }
    }

    // General case, a path or a local:
    let path = name_ref.syntax().ancestors().find_map(ast::Path::cast)?;
    let resolved = analyzer.resolve_path(db, &path)?;
    match resolved {
        PathResolution::Def(def) => Some(def.definition(db)),
        PathResolution::LocalBinding(Either::A(pat)) => decl_from_pat(db, file_id, pat),
        PathResolution::LocalBinding(Either::B(par)) => {
            Some(Definition { item: NameKind::SelfParam(par), container, visibility })
        }
        PathResolution::GenericParam(par) => {
            // FIXME: get generic param def
            Some(Definition { item: NameKind::GenericParam(par), container, visibility })
        }
        PathResolution::Macro(def) => {
            Some(Definition { item: NameKind::Macro(def), container, visibility })
        }
        PathResolution::SelfType(impl_block) => {
            let ty = impl_block.target_ty(db);
            let container = impl_block.module();
            Some(Definition { item: NameKind::SelfType(ty), container, visibility })
        }
        PathResolution::AssocItem(assoc) => Some(assoc.definition(db)),
    }
}

fn decl_from_pat(
    db: &RootDatabase,
    file_id: HirFileId,
    pat: AstPtr<ast::BindPat>,
) -> Option<Definition> {
    let root = db.parse_or_expand(file_id)?;
    // FIXME: use match_ast!
    let def = pat.to_node(&root).syntax().ancestors().find_map(|node| {
        if let Some(it) = ast::FnDef::cast(node.clone()) {
            let src = hir::Source { file_id, ast: it };
            Some(hir::Function::from_source(db, src)?.into())
        } else if let Some(it) = ast::ConstDef::cast(node.clone()) {
            let src = hir::Source { file_id, ast: it };
            Some(hir::Const::from_source(db, src)?.into())
        } else if let Some(it) = ast::StaticDef::cast(node.clone()) {
            let src = hir::Source { file_id, ast: it };
            Some(hir::Static::from_source(db, src)?.into())
        } else {
            None
        }
    })?;
    let item = NameKind::Pat((def, pat));
    let container = def.module(db);
    Some(Definition { item, container, visibility: None })
}
