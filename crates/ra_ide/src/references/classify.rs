//! Functions that are used to classify an element from its definition or reference.

use hir::{InFile, PathResolution, SourceBinder};
use ra_prof::profile;
use ra_syntax::{ast, match_ast, AstNode};
use test_utils::tested_by;

use super::{
    name_definition::{from_assoc_item, from_module_def, from_struct_field},
    NameDefinition, NameKind,
};
use crate::db::RootDatabase;

pub(crate) fn classify_name(
    sb: &mut SourceBinder<RootDatabase>,
    name: InFile<&ast::Name>,
) -> Option<NameDefinition> {
    let _p = profile("classify_name");
    let parent = name.value.syntax().parent()?;

    match_ast! {
        match parent {
            ast::BindPat(it) => {
                let src = name.with_value(it);
                let local = sb.to_def(src)?;
                Some(NameDefinition {
                    visibility: None,
                    container: local.module(sb.db),
                    kind: NameKind::Local(local),
                })
            },
            ast::RecordFieldDef(it) => {
                let src = name.with_value(it);
                let field: hir::StructField = sb.to_def(src)?;
                Some(from_struct_field(sb.db, field))
            },
            ast::Module(it) => {
                let def = sb.to_def(name.with_value(it))?;
                Some(from_module_def(sb.db, def.into(), None))
            },
            ast::StructDef(it) => {
                let src = name.with_value(it);
                let def: hir::Struct = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into(), None))
            },
            ast::EnumDef(it) => {
                let src = name.with_value(it);
                let def: hir::Enum = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into(), None))
            },
            ast::TraitDef(it) => {
                let src = name.with_value(it);
                let def: hir::Trait = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into(), None))
            },
            ast::StaticDef(it) => {
                let src = name.with_value(it);
                let def: hir::Static = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into(), None))
            },
            ast::EnumVariant(it) => {
                let src = name.with_value(it);
                let def: hir::EnumVariant = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into(), None))
            },
            ast::FnDef(it) => {
                let src = name.with_value(it);
                let def: hir::Function = sb.to_def(src)?;
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    Some(from_assoc_item(sb.db, def.into()))
                } else {
                    Some(from_module_def(sb.db, def.into(), None))
                }
            },
            ast::ConstDef(it) => {
                let src = name.with_value(it);
                let def: hir::Const = sb.to_def(src)?;
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    Some(from_assoc_item(sb.db, def.into()))
                } else {
                    Some(from_module_def(sb.db, def.into(), None))
                }
            },
            ast::TypeAliasDef(it) => {
                let src = name.with_value(it);
                let def: hir::TypeAlias = sb.to_def(src)?;
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    Some(from_assoc_item(sb.db, def.into()))
                } else {
                    Some(from_module_def(sb.db, def.into(), None))
                }
            },
            ast::MacroCall(it) => {
                let src = name.with_value(it);
                let def = sb.to_def(src.clone())?;

                let module = sb.to_module_def(src.file_id.original_file(sb.db))?;

                Some(NameDefinition {
                    visibility: None,
                    container: module,
                    kind: NameKind::Macro(def),
                })
            },
            ast::TypeParam(it) => {
                let src = name.with_value(it);
                let def = sb.to_def(src)?;
                Some(NameDefinition {
                    visibility: None,
                    container: def.module(sb.db),
                    kind: NameKind::TypeParam(def),
                })
            },
            _ => None,
        }
    }
}

pub(crate) fn classify_name_ref(
    sb: &mut SourceBinder<RootDatabase>,
    name_ref: InFile<&ast::NameRef>,
) -> Option<NameDefinition> {
    let _p = profile("classify_name_ref");

    let parent = name_ref.value.syntax().parent()?;
    let analyzer = sb.analyze(name_ref.map(|it| it.syntax()), None);

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_methods);
        if let Some(func) = analyzer.resolve_method_call(&method_call) {
            return Some(from_assoc_item(sb.db, func.into()));
        }
    }

    if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_fields);
        if let Some(field) = analyzer.resolve_field(&field_expr) {
            return Some(from_struct_field(sb.db, field));
        }
    }

    if let Some(record_field) = ast::RecordField::cast(parent.clone()) {
        tested_by!(goto_def_for_record_fields);
        tested_by!(goto_def_for_field_init_shorthand);
        if let Some(field_def) = analyzer.resolve_record_field(&record_field) {
            return Some(from_struct_field(sb.db, field_def));
        }
    }

    // FIXME: find correct container and visibility for each case
    let visibility = None;
    let container = sb.to_module_def(name_ref.file_id.original_file(sb.db))?;

    if let Some(macro_call) = parent.ancestors().find_map(ast::MacroCall::cast) {
        tested_by!(goto_def_for_macros);
        if let Some(macro_def) =
            analyzer.resolve_macro_call(sb.db, name_ref.with_value(&macro_call))
        {
            let kind = NameKind::Macro(macro_def);
            return Some(NameDefinition { kind, container, visibility });
        }
    }

    let path = name_ref.value.syntax().ancestors().find_map(ast::Path::cast)?;
    let resolved = analyzer.resolve_path(sb.db, &path)?;
    match resolved {
        PathResolution::Def(def) => Some(from_module_def(sb.db, def, Some(container))),
        PathResolution::AssocItem(item) => Some(from_assoc_item(sb.db, item)),
        PathResolution::Local(local) => {
            let kind = NameKind::Local(local);
            let container = local.module(sb.db);
            Some(NameDefinition { kind, container, visibility: None })
        }
        PathResolution::TypeParam(par) => {
            let kind = NameKind::TypeParam(par);
            let container = par.module(sb.db);
            Some(NameDefinition { kind, container, visibility })
        }
        PathResolution::Macro(def) => {
            let kind = NameKind::Macro(def);
            Some(NameDefinition { kind, container, visibility })
        }
        PathResolution::SelfType(impl_block) => {
            let kind = NameKind::SelfType(impl_block);
            let container = impl_block.module(sb.db);
            Some(NameDefinition { kind, container, visibility })
        }
    }
}
