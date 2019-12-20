//! Functions that are used to classify an element from its definition or reference.

use hir::{FromSource, InFile, Module, ModuleSource, PathResolution, SourceAnalyzer};
use ra_prof::profile;
use ra_syntax::{ast, match_ast, AstNode};
use test_utils::tested_by;

use super::{
    name_definition::{from_assoc_item, from_module_def, from_struct_field},
    NameDefinition, NameKind,
};
use crate::db::RootDatabase;

pub(crate) fn classify_name(db: &RootDatabase, name: InFile<&ast::Name>) -> Option<NameDefinition> {
    let _p = profile("classify_name");
    let parent = name.value.syntax().parent()?;

    match_ast! {
        match parent {
            ast::BindPat(it) => {
                let src = name.with_value(it);
                let local = hir::Local::from_source(db, src)?;
                Some(NameDefinition {
                    visibility: None,
                    container: local.module(db),
                    kind: NameKind::Local(local),
                })
            },
            ast::RecordFieldDef(it) => {
                let ast = hir::FieldSource::Named(it);
                let src = name.with_value(ast);
                let field = hir::StructField::from_source(db, src)?;
                Some(from_struct_field(db, field))
            },
            ast::Module(it) => {
                let def = {
                    if !it.has_semi() {
                        let ast = hir::ModuleSource::Module(it);
                        let src = name.with_value(ast);
                        hir::Module::from_definition(db, src)
                    } else {
                        let src = name.with_value(it);
                        hir::Module::from_declaration(db, src)
                    }
                }?;
                Some(from_module_def(db, def.into(), None))
            },
            ast::StructDef(it) => {
                let src = name.with_value(it);
                let def = hir::Struct::from_source(db, src)?;
                Some(from_module_def(db, def.into(), None))
            },
            ast::EnumDef(it) => {
                let src = name.with_value(it);
                let def = hir::Enum::from_source(db, src)?;
                Some(from_module_def(db, def.into(), None))
            },
            ast::TraitDef(it) => {
                let src = name.with_value(it);
                let def = hir::Trait::from_source(db, src)?;
                Some(from_module_def(db, def.into(), None))
            },
            ast::StaticDef(it) => {
                let src = name.with_value(it);
                let def = hir::Static::from_source(db, src)?;
                Some(from_module_def(db, def.into(), None))
            },
            ast::EnumVariant(it) => {
                let src = name.with_value(it);
                let def = hir::EnumVariant::from_source(db, src)?;
                Some(from_module_def(db, def.into(), None))
            },
            ast::FnDef(it) => {
                let src = name.with_value(it);
                let def = hir::Function::from_source(db, src)?;
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    Some(from_assoc_item(db, def.into()))
                } else {
                    Some(from_module_def(db, def.into(), None))
                }
            },
            ast::ConstDef(it) => {
                let src = name.with_value(it);
                let def = hir::Const::from_source(db, src)?;
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    Some(from_assoc_item(db, def.into()))
                } else {
                    Some(from_module_def(db, def.into(), None))
                }
            },
            ast::TypeAliasDef(it) => {
                let src = name.with_value(it);
                let def = hir::TypeAlias::from_source(db, src)?;
                if parent.parent().and_then(ast::ItemList::cast).is_some() {
                    Some(from_assoc_item(db, def.into()))
                } else {
                    Some(from_module_def(db, def.into(), None))
                }
            },
            ast::MacroCall(it) => {
                let src = name.with_value(it);
                let def = hir::MacroDef::from_source(db, src.clone())?;

                let module_src = ModuleSource::from_child_node(db, src.as_ref().map(|it| it.syntax()));
                let module = Module::from_definition(db, src.with_value(module_src))?;

                Some(NameDefinition {
                    visibility: None,
                    container: module,
                    kind: NameKind::Macro(def),
                })
            },
            ast::TypeParam(it) => {
                let src = name.with_value(it);
                let def = hir::TypeParam::from_source(db, src)?;
                Some(NameDefinition {
                    visibility: None,
                    container: def.module(db),
                    kind: NameKind::TypeParam(def),
                })
            },
            _ => None,
        }
    }
}

pub(crate) fn classify_name_ref(
    db: &RootDatabase,
    name_ref: InFile<&ast::NameRef>,
) -> Option<NameDefinition> {
    let _p = profile("classify_name_ref");

    let parent = name_ref.value.syntax().parent()?;
    let analyzer = SourceAnalyzer::new(db, name_ref.map(|it| it.syntax()), None);

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_methods);
        if let Some(func) = analyzer.resolve_method_call(&method_call) {
            return Some(from_assoc_item(db, func.into()));
        }
    }

    if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_fields);
        if let Some(field) = analyzer.resolve_field(&field_expr) {
            return Some(from_struct_field(db, field));
        }
    }

    if let Some(record_field) = ast::RecordField::cast(parent.clone()) {
        tested_by!(goto_def_for_record_fields);
        tested_by!(goto_def_for_field_init_shorthand);
        if let Some(field_def) = analyzer.resolve_record_field(&record_field) {
            return Some(from_struct_field(db, field_def));
        }
    }

    let ast = ModuleSource::from_child_node(db, name_ref.with_value(&parent));
    // FIXME: find correct container and visibility for each case
    let container = Module::from_definition(db, name_ref.with_value(ast))?;
    let visibility = None;

    if let Some(macro_call) = parent.ancestors().find_map(ast::MacroCall::cast) {
        tested_by!(goto_def_for_macros);
        if let Some(macro_def) = analyzer.resolve_macro_call(db, name_ref.with_value(&macro_call)) {
            let kind = NameKind::Macro(macro_def);
            return Some(NameDefinition { kind, container, visibility });
        }
    }

    let path = name_ref.value.syntax().ancestors().find_map(ast::Path::cast)?;
    let resolved = analyzer.resolve_path(db, &path)?;
    match resolved {
        PathResolution::Def(def) => Some(from_module_def(db, def, Some(container))),
        PathResolution::AssocItem(item) => Some(from_assoc_item(db, item)),
        PathResolution::Local(local) => {
            let container = local.module(db);
            let kind = NameKind::Local(local);
            Some(NameDefinition { kind, container, visibility: None })
        }
        PathResolution::TypeParam(par) => {
            let kind = NameKind::TypeParam(par);
            Some(NameDefinition { kind, container, visibility })
        }
        PathResolution::Macro(def) => {
            let kind = NameKind::Macro(def);
            Some(NameDefinition { kind, container, visibility })
        }
        PathResolution::SelfType(impl_block) => {
            let kind = NameKind::SelfType(impl_block);
            let container = impl_block.module(db);
            Some(NameDefinition { kind, container, visibility })
        }
    }
}
