//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use hir::{
    Adt, FieldSource, HasSource, ImplBlock, InFile, Local, MacroDef, Module, ModuleDef,
    SourceBinder, StructField, TypeParam,
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

    pub fn visibility(&self, db: &RootDatabase) -> Option<ast::Visibility> {
        match self.kind {
            NameKind::Macro(_) => None,
            NameKind::StructField(sf) => match sf.source(db).value {
                FieldSource::Named(it) => it.visibility(),
                FieldSource::Pos(it) => it.visibility(),
            },
            NameKind::ModuleDef(def) => match def {
                ModuleDef::Module(it) => it.declaration_source(db)?.value.visibility(),
                ModuleDef::Function(it) => it.source(db).value.visibility(),
                ModuleDef::Adt(adt) => match adt {
                    Adt::Struct(it) => it.source(db).value.visibility(),
                    Adt::Union(it) => it.source(db).value.visibility(),
                    Adt::Enum(it) => it.source(db).value.visibility(),
                },
                ModuleDef::Const(it) => it.source(db).value.visibility(),
                ModuleDef::Static(it) => it.source(db).value.visibility(),
                ModuleDef::Trait(it) => it.source(db).value.visibility(),
                ModuleDef::TypeAlias(it) => it.source(db).value.visibility(),
                ModuleDef::EnumVariant(_) => None,
                ModuleDef::BuiltinType(_) => None,
            },
            NameKind::SelfType(_) => None,
            NameKind::Local(_) => None,
            NameKind::TypeParam(_) => None,
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
                    kind: NameKind::Local(local),
                })
            },
            ast::RecordFieldDef(it) => {
                let src = name.with_value(it);
                let field: hir::StructField = sb.to_def(src)?;
                Some(from_struct_field(field))
            },
            ast::Module(it) => {
                let def = sb.to_def(name.with_value(it))?;
                Some(from_module_def(def.into()))
            },
            ast::StructDef(it) => {
                let src = name.with_value(it);
                let def: hir::Struct = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::EnumDef(it) => {
                let src = name.with_value(it);
                let def: hir::Enum = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::TraitDef(it) => {
                let src = name.with_value(it);
                let def: hir::Trait = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::StaticDef(it) => {
                let src = name.with_value(it);
                let def: hir::Static = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::EnumVariant(it) => {
                let src = name.with_value(it);
                let def: hir::EnumVariant = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::FnDef(it) => {
                let src = name.with_value(it);
                let def: hir::Function = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::ConstDef(it) => {
                let src = name.with_value(it);
                let def: hir::Const = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::TypeAliasDef(it) => {
                let src = name.with_value(it);
                let def: hir::TypeAlias = sb.to_def(src)?;
                Some(from_module_def(def.into()))
            },
            ast::MacroCall(it) => {
                let src = name.with_value(it);
                let def = sb.to_def(src.clone())?;

                Some(NameDefinition {
                    kind: NameKind::Macro(def),
                })
            },
            ast::TypeParam(it) => {
                let src = name.with_value(it);
                let def = sb.to_def(src)?;
                Some(NameDefinition {
                    kind: NameKind::TypeParam(def),
                })
            },
            _ => None,
        }
    }
}

pub fn from_struct_field(field: StructField) -> NameDefinition {
    let kind = NameKind::StructField(field);
    NameDefinition { kind }
}

pub fn from_module_def(def: ModuleDef) -> NameDefinition {
    let kind = NameKind::ModuleDef(def);
    NameDefinition { kind }
}
