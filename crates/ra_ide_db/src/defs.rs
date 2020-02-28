//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use hir::{
    Adt, FieldSource, HasSource, ImplBlock, Local, MacroDef, Module, ModuleDef, Semantics,
    StructField, TypeParam,
};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode, VisibilityOwner},
    match_ast,
};

use crate::RootDatabase;

#[derive(Debug, PartialEq, Eq)]
pub enum NameDefinition {
    Macro(MacroDef),
    StructField(StructField),
    ModuleDef(ModuleDef),
    SelfType(ImplBlock),
    Local(Local),
    TypeParam(TypeParam),
}

impl NameDefinition {
    pub fn module(&self, db: &RootDatabase) -> Option<Module> {
        match self {
            NameDefinition::Macro(it) => it.module(db),
            NameDefinition::StructField(it) => Some(it.parent_def(db).module(db)),
            NameDefinition::ModuleDef(it) => it.module(db),
            NameDefinition::SelfType(it) => Some(it.module(db)),
            NameDefinition::Local(it) => Some(it.module(db)),
            NameDefinition::TypeParam(it) => Some(it.module(db)),
        }
    }

    pub fn visibility(&self, db: &RootDatabase) -> Option<ast::Visibility> {
        match self {
            NameDefinition::Macro(_) => None,
            NameDefinition::StructField(sf) => match sf.source(db).value {
                FieldSource::Named(it) => it.visibility(),
                FieldSource::Pos(it) => it.visibility(),
            },
            NameDefinition::ModuleDef(def) => match def {
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
            NameDefinition::SelfType(_) => None,
            NameDefinition::Local(_) => None,
            NameDefinition::TypeParam(_) => None,
        }
    }
}

pub enum NameClass {
    NameDefinition(NameDefinition),
    /// `None` in `if let None = Some(82) {}`
    ConstReference(NameDefinition),
}

impl NameClass {
    pub fn into_definition(self) -> Option<NameDefinition> {
        match self {
            NameClass::NameDefinition(it) => Some(it),
            NameClass::ConstReference(_) => None,
        }
    }

    pub fn definition(self) -> NameDefinition {
        match self {
            NameClass::NameDefinition(it) | NameClass::ConstReference(it) => it,
        }
    }
}

pub fn classify_name(sema: &Semantics<RootDatabase>, name: &ast::Name) -> Option<NameClass> {
    classify_name_inner(sema, name).map(NameClass::NameDefinition)
}

fn classify_name_inner(sema: &Semantics<RootDatabase>, name: &ast::Name) -> Option<NameDefinition> {
    let _p = profile("classify_name");
    let parent = name.syntax().parent()?;

    match_ast! {
        match parent {
            ast::BindPat(it) => {
                let local = sema.to_def(&it)?;
                Some(NameDefinition::Local(local))
            },
            ast::RecordFieldDef(it) => {
                let field: hir::StructField = sema.to_def(&it)?;
                Some(from_struct_field(field))
            },
            ast::Module(it) => {
                let def = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::StructDef(it) => {
                let def: hir::Struct = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::UnionDef(it) => {
                let def: hir::Union = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::EnumDef(it) => {
                let def: hir::Enum = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::TraitDef(it) => {
                let def: hir::Trait = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::StaticDef(it) => {
                let def: hir::Static = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::EnumVariant(it) => {
                let def: hir::EnumVariant = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::FnDef(it) => {
                let def: hir::Function = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::ConstDef(it) => {
                let def: hir::Const = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::TypeAliasDef(it) => {
                let def: hir::TypeAlias = sema.to_def(&it)?;
                Some(from_module_def(def.into()))
            },
            ast::MacroCall(it) => {
                let def = sema.to_def(&it)?;
                Some(NameDefinition::Macro(def))
            },
            ast::TypeParam(it) => {
                let def = sema.to_def(&it)?;
                Some(NameDefinition::TypeParam(def))
            },
            _ => None,
        }
    }
}

pub fn from_struct_field(field: StructField) -> NameDefinition {
    NameDefinition::StructField(field)
}

pub fn from_module_def(def: ModuleDef) -> NameDefinition {
    NameDefinition::ModuleDef(def)
}
