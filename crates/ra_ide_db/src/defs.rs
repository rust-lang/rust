//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use hir::{
    Adt, HasSource, ImplBlock, InFile, Local, MacroDef, Module, ModuleDef, SourceBinder,
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
    StructField(StructField),
    ModuleDef(ModuleDef),
    SelfType(ImplBlock),
    Local(Local),
    TypeParam(TypeParam),
}

#[derive(PartialEq, Eq)]
pub struct NameDefinition {
    pub visibility: Option<ast::Visibility>,
    /// FIXME: this doesn't really make sense. For example, builtin types don't
    /// really have a module.
    pub kind: NameKind,
}

impl NameDefinition {
    pub fn module(&self, db: &RootDatabase) -> Option<Module> {
        match self.kind {
            NameKind::Macro(it) => it.module(db),
            NameKind::StructField(it) => Some(it.parent_def(db).module(db)),
            NameKind::ModuleDef(it) => it.module(db),
            NameKind::SelfType(it) => Some(it.module(db)),
            NameKind::Local(it) => Some(it.module(db)),
            NameKind::TypeParam(it) => Some(it.module(db)),
        }
    }
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
                Some(from_module_def(sb.db, def.into()))
            },
            ast::StructDef(it) => {
                let src = name.with_value(it);
                let def: hir::Struct = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::EnumDef(it) => {
                let src = name.with_value(it);
                let def: hir::Enum = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::TraitDef(it) => {
                let src = name.with_value(it);
                let def: hir::Trait = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::StaticDef(it) => {
                let src = name.with_value(it);
                let def: hir::Static = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::EnumVariant(it) => {
                let src = name.with_value(it);
                let def: hir::EnumVariant = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::FnDef(it) => {
                let src = name.with_value(it);
                let def: hir::Function = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::ConstDef(it) => {
                let src = name.with_value(it);
                let def: hir::Const = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::TypeAliasDef(it) => {
                let src = name.with_value(it);
                let def: hir::TypeAlias = sb.to_def(src)?;
                Some(from_module_def(sb.db, def.into()))
            },
            ast::MacroCall(it) => {
                let src = name.with_value(it);
                let def = sb.to_def(src.clone())?;

                Some(NameDefinition {
                    visibility: None,
                    kind: NameKind::Macro(def),
                })
            },
            ast::TypeParam(it) => {
                let src = name.with_value(it);
                let def = sb.to_def(src)?;
                Some(NameDefinition {
                    visibility: None,
                    kind: NameKind::TypeParam(def),
                })
            },
            _ => None,
        }
    }
}

pub fn from_struct_field(db: &RootDatabase, field: StructField) -> NameDefinition {
    let kind = NameKind::StructField(field);
    let parent = field.parent_def(db);
    let visibility = match parent {
        VariantDef::Struct(s) => s.source(db).value.visibility(),
        VariantDef::Union(e) => e.source(db).value.visibility(),
        VariantDef::EnumVariant(e) => e.source(db).value.parent_enum().visibility(),
    };
    NameDefinition { kind, visibility }
}

pub fn from_module_def(db: &RootDatabase, def: ModuleDef) -> NameDefinition {
    let kind = NameKind::ModuleDef(def);
    let visibility = match def {
        ModuleDef::Module(it) => it.declaration_source(db).and_then(|s| s.value.visibility()),
        ModuleDef::EnumVariant(it) => it.source(db).value.parent_enum().visibility(),
        ModuleDef::Function(it) => it.source(db).value.visibility(),
        ModuleDef::Const(it) => it.source(db).value.visibility(),
        ModuleDef::Static(it) => it.source(db).value.visibility(),
        ModuleDef::Trait(it) => it.source(db).value.visibility(),
        ModuleDef::TypeAlias(it) => it.source(db).value.visibility(),
        ModuleDef::Adt(Adt::Struct(it)) => it.source(db).value.visibility(),
        ModuleDef::Adt(Adt::Union(it)) => it.source(db).value.visibility(),
        ModuleDef::Adt(Adt::Enum(it)) => it.source(db).value.visibility(),
        ModuleDef::BuiltinType(..) => None,
    };
    NameDefinition { kind, visibility }
}
