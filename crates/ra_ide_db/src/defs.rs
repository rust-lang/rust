//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use hir::{
    Adt, FieldSource, HasSource, ImplDef, Local, MacroDef, Module, ModuleDef, Name, PathResolution,
    Semantics, StructField, TypeParam,
};
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode, VisibilityOwner},
    match_ast,
};
use test_utils::tested_by;

use crate::RootDatabase;

// FIXME: a more precise name would probably be `Symbol`?
#[derive(Debug, PartialEq, Eq)]
pub enum Definition {
    Macro(MacroDef),
    StructField(StructField),
    ModuleDef(ModuleDef),
    SelfType(ImplDef),
    Local(Local),
    TypeParam(TypeParam),
}

impl Definition {
    pub fn module(&self, db: &RootDatabase) -> Option<Module> {
        match self {
            Definition::Macro(it) => it.module(db),
            Definition::StructField(it) => Some(it.parent_def(db).module(db)),
            Definition::ModuleDef(it) => it.module(db),
            Definition::SelfType(it) => Some(it.module(db)),
            Definition::Local(it) => Some(it.module(db)),
            Definition::TypeParam(it) => Some(it.module(db)),
        }
    }

    pub fn visibility(&self, db: &RootDatabase) -> Option<ast::Visibility> {
        match self {
            Definition::Macro(_) => None,
            Definition::StructField(sf) => match sf.source(db).value {
                FieldSource::Named(it) => it.visibility(),
                FieldSource::Pos(it) => it.visibility(),
            },
            Definition::ModuleDef(def) => match def {
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
            Definition::SelfType(_) => None,
            Definition::Local(_) => None,
            Definition::TypeParam(_) => None,
        }
    }

    pub fn name(&self, db: &RootDatabase) -> Option<Name> {
        let name = match self {
            Definition::Macro(it) => it.name(db)?,
            Definition::StructField(it) => it.name(db),
            Definition::ModuleDef(def) => match def {
                hir::ModuleDef::Module(it) => it.name(db)?,
                hir::ModuleDef::Function(it) => it.name(db),
                hir::ModuleDef::Adt(def) => match def {
                    hir::Adt::Struct(it) => it.name(db),
                    hir::Adt::Union(it) => it.name(db),
                    hir::Adt::Enum(it) => it.name(db),
                },
                hir::ModuleDef::EnumVariant(it) => it.name(db),
                hir::ModuleDef::Const(it) => it.name(db)?,
                hir::ModuleDef::Static(it) => it.name(db)?,
                hir::ModuleDef::Trait(it) => it.name(db),
                hir::ModuleDef::TypeAlias(it) => it.name(db),
                hir::ModuleDef::BuiltinType(_) => return None,
            },
            Definition::SelfType(_) => return None,
            Definition::Local(it) => it.name(db)?,
            Definition::TypeParam(it) => it.name(db),
        };
        Some(name)
    }
}

pub enum NameClass {
    Definition(Definition),
    /// `None` in `if let None = Some(82) {}`
    ConstReference(Definition),
}

impl NameClass {
    pub fn into_definition(self) -> Option<Definition> {
        match self {
            NameClass::Definition(it) => Some(it),
            NameClass::ConstReference(_) => None,
        }
    }

    pub fn definition(self) -> Definition {
        match self {
            NameClass::Definition(it) | NameClass::ConstReference(it) => it,
        }
    }
}

pub fn classify_name(sema: &Semantics<RootDatabase>, name: &ast::Name) -> Option<NameClass> {
    let _p = profile("classify_name");

    if let Some(bind_pat) = name.syntax().parent().and_then(ast::BindPat::cast) {
        if let Some(def) = sema.resolve_bind_pat_to_const(&bind_pat) {
            return Some(NameClass::ConstReference(Definition::ModuleDef(def)));
        }
    }

    classify_name_inner(sema, name).map(NameClass::Definition)
}

fn classify_name_inner(sema: &Semantics<RootDatabase>, name: &ast::Name) -> Option<Definition> {
    let parent = name.syntax().parent()?;

    match_ast! {
        match parent {
            ast::BindPat(it) => {
                let local = sema.to_def(&it)?;
                Some(Definition::Local(local))
            },
            ast::RecordFieldDef(it) => {
                let field: hir::StructField = sema.to_def(&it)?;
                Some(Definition::StructField(field))
            },
            ast::Module(it) => {
                let def = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::StructDef(it) => {
                let def: hir::Struct = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::UnionDef(it) => {
                let def: hir::Union = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::EnumDef(it) => {
                let def: hir::Enum = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::TraitDef(it) => {
                let def: hir::Trait = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::StaticDef(it) => {
                let def: hir::Static = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::EnumVariant(it) => {
                let def: hir::EnumVariant = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::FnDef(it) => {
                let def: hir::Function = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::ConstDef(it) => {
                let def: hir::Const = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::TypeAliasDef(it) => {
                let def: hir::TypeAlias = sema.to_def(&it)?;
                Some(Definition::ModuleDef(def.into()))
            },
            ast::MacroCall(it) => {
                let def = sema.to_def(&it)?;
                Some(Definition::Macro(def))
            },
            ast::TypeParam(it) => {
                let def = sema.to_def(&it)?;
                Some(Definition::TypeParam(def))
            },
            _ => None,
        }
    }
}

pub enum NameRefClass {
    Definition(Definition),
    FieldShorthand { local: Local, field: Definition },
}

impl NameRefClass {
    pub fn definition(self) -> Definition {
        match self {
            NameRefClass::Definition(def) => def,
            NameRefClass::FieldShorthand { local, field: _ } => Definition::Local(local),
        }
    }
}

pub fn classify_name_ref(
    sema: &Semantics<RootDatabase>,
    name_ref: &ast::NameRef,
) -> Option<NameRefClass> {
    let _p = profile("classify_name_ref");

    let parent = name_ref.syntax().parent()?;

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_methods; force);
        if let Some(func) = sema.resolve_method_call(&method_call) {
            return Some(NameRefClass::Definition(Definition::ModuleDef(func.into())));
        }
    }

    if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_fields; force);
        if let Some(field) = sema.resolve_field(&field_expr) {
            return Some(NameRefClass::Definition(Definition::StructField(field)));
        }
    }

    if let Some(record_field) = ast::RecordField::cast(parent.clone()) {
        tested_by!(goto_def_for_record_fields; force);
        tested_by!(goto_def_for_field_init_shorthand; force);
        if let Some((field, local)) = sema.resolve_record_field(&record_field) {
            let field = Definition::StructField(field);
            let res = match local {
                None => NameRefClass::Definition(field),
                Some(local) => NameRefClass::FieldShorthand { field, local },
            };
            return Some(res);
        }
    }

    if let Some(macro_call) = parent.ancestors().find_map(ast::MacroCall::cast) {
        tested_by!(goto_def_for_macros; force);
        if let Some(macro_def) = sema.resolve_macro_call(&macro_call) {
            return Some(NameRefClass::Definition(Definition::Macro(macro_def)));
        }
    }

    let path = name_ref.syntax().ancestors().find_map(ast::Path::cast)?;
    let resolved = sema.resolve_path(&path)?;
    let res = match resolved {
        PathResolution::Def(def) => Definition::ModuleDef(def),
        PathResolution::AssocItem(item) => {
            let def = match item {
                hir::AssocItem::Function(it) => it.into(),
                hir::AssocItem::Const(it) => it.into(),
                hir::AssocItem::TypeAlias(it) => it.into(),
            };
            Definition::ModuleDef(def)
        }
        PathResolution::Local(local) => Definition::Local(local),
        PathResolution::TypeParam(par) => Definition::TypeParam(par),
        PathResolution::Macro(def) => Definition::Macro(def),
        PathResolution::SelfType(impl_def) => Definition::SelfType(impl_def),
    };
    Some(NameRefClass::Definition(res))
}
