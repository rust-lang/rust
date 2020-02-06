//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use hir::{
    Adt, AssocItem, HasSource, ImplBlock, InFile, Local, MacroDef, Module, ModuleDef, SourceBinder,
    StructField, TypeParam, VariantDef,
};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode, VisibilityOwner},
    match_ast,
};

use crate::RootDatabase;

#[derive(Debug, PartialEq, Eq)]
pub enum NameKind {
    Macro(MacroDef),
    Field(StructField),
    AssocItem(AssocItem),
    Def(ModuleDef),
    SelfType(ImplBlock),
    Local(Local),
    TypeParam(TypeParam),
}

#[derive(PartialEq, Eq)]
pub struct NameDefinition {
    pub visibility: Option<ast::Visibility>,
    /// FIXME: this doesn't really make sense. For example, builtin types don't
    /// really have a module.
    pub container: Module,
    pub kind: NameKind,
}

pub fn classify_name(
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
                if parent.parent().and_then(ast::ItemList::cast).map_or(false, |it| it.syntax().parent().and_then(ast::Module::cast).is_none()) {
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

pub fn from_assoc_item(db: &RootDatabase, item: AssocItem) -> NameDefinition {
    let container = item.module(db);
    let visibility = match item {
        AssocItem::Function(f) => f.source(db).value.visibility(),
        AssocItem::Const(c) => c.source(db).value.visibility(),
        AssocItem::TypeAlias(a) => a.source(db).value.visibility(),
    };
    let kind = NameKind::AssocItem(item);
    NameDefinition { kind, container, visibility }
}

pub fn from_struct_field(db: &RootDatabase, field: StructField) -> NameDefinition {
    let kind = NameKind::Field(field);
    let parent = field.parent_def(db);
    let container = parent.module(db);
    let visibility = match parent {
        VariantDef::Struct(s) => s.source(db).value.visibility(),
        VariantDef::Union(e) => e.source(db).value.visibility(),
        VariantDef::EnumVariant(e) => e.source(db).value.parent_enum().visibility(),
    };
    NameDefinition { kind, container, visibility }
}

pub fn from_module_def(
    db: &RootDatabase,
    def: ModuleDef,
    module: Option<Module>,
) -> NameDefinition {
    let kind = NameKind::Def(def);
    let (container, visibility) = match def {
        ModuleDef::Module(it) => {
            let container = it.parent(db).or_else(|| Some(it)).unwrap();
            let visibility = it.declaration_source(db).and_then(|s| s.value.visibility());
            (container, visibility)
        }
        ModuleDef::EnumVariant(it) => {
            let container = it.module(db);
            let visibility = it.source(db).value.parent_enum().visibility();
            (container, visibility)
        }
        ModuleDef::Function(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Const(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Static(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Trait(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::TypeAlias(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Adt(Adt::Struct(it)) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Adt(Adt::Union(it)) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Adt(Adt::Enum(it)) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::BuiltinType(..) => (module.unwrap(), None),
    };
    NameDefinition { kind, container, visibility }
}
