//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use hir::{
    db::HirDatabase, Crate, Field, HasVisibility, ImplDef, Local, MacroDef, Module, ModuleDef,
    Name, PathResolution, Semantics, TypeParam, Visibility,
};
use syntax::{
    ast::{self, AstNode},
    match_ast, SyntaxNode,
};

use crate::RootDatabase;

// FIXME: a more precise name would probably be `Symbol`?
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Definition {
    Macro(MacroDef),
    Field(Field),
    ModuleDef(ModuleDef),
    SelfType(ImplDef),
    Local(Local),
    TypeParam(TypeParam),
}

impl Definition {
    pub fn module(&self, db: &RootDatabase) -> Option<Module> {
        match self {
            Definition::Macro(it) => it.module(db),
            Definition::Field(it) => Some(it.parent_def(db).module(db)),
            Definition::ModuleDef(it) => it.module(db),
            Definition::SelfType(it) => Some(it.module(db)),
            Definition::Local(it) => Some(it.module(db)),
            Definition::TypeParam(it) => Some(it.module(db)),
        }
    }

    pub fn visibility(&self, db: &RootDatabase) -> Option<Visibility> {
        match self {
            Definition::Macro(_) => None,
            Definition::Field(sf) => Some(sf.visibility(db)),
            Definition::ModuleDef(def) => def.definition_visibility(db),
            Definition::SelfType(_) => None,
            Definition::Local(_) => None,
            Definition::TypeParam(_) => None,
        }
    }

    pub fn name(&self, db: &RootDatabase) -> Option<Name> {
        let name = match self {
            Definition::Macro(it) => it.name(db)?,
            Definition::Field(it) => it.name(db),
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

#[derive(Debug)]
pub enum NameClass {
    ExternCrate(Crate),
    Definition(Definition),
    /// `None` in `if let None = Some(82) {}`
    ConstReference(Definition),
    FieldShorthand {
        local: Local,
        field: Definition,
    },
}

impl NameClass {
    pub fn into_definition(self, db: &dyn HirDatabase) -> Option<Definition> {
        Some(match self {
            NameClass::ExternCrate(krate) => Definition::ModuleDef(krate.root_module(db).into()),
            NameClass::Definition(it) => it,
            NameClass::ConstReference(_) => return None,
            NameClass::FieldShorthand { local, field: _ } => Definition::Local(local),
        })
    }

    pub fn definition(self, db: &dyn HirDatabase) -> Definition {
        match self {
            NameClass::ExternCrate(krate) => Definition::ModuleDef(krate.root_module(db).into()),
            NameClass::Definition(it) | NameClass::ConstReference(it) => it,
            NameClass::FieldShorthand { local: _, field } => field,
        }
    }
}

pub fn classify_name(sema: &Semantics<RootDatabase>, name: &ast::Name) -> Option<NameClass> {
    let _p = profile::span("classify_name");

    let parent = name.syntax().parent()?;

    if let Some(bind_pat) = ast::IdentPat::cast(parent.clone()) {
        if let Some(def) = sema.resolve_bind_pat_to_const(&bind_pat) {
            return Some(NameClass::ConstReference(Definition::ModuleDef(def)));
        }
    }

    match_ast! {
        match parent {
            ast::Rename(it) => {
                if let Some(use_tree) = it.syntax().parent().and_then(ast::UseTree::cast) {
                    let path = use_tree.path()?;
                    let path_segment = path.segment()?;
                    let name_ref_class = path_segment
                        .name_ref()
                        // The rename might be from a `self` token, so fallback to the name higher
                        // in the use tree.
                        .or_else(||{
                            if path_segment.self_token().is_none() {
                                return None;
                            }

                            let use_tree = use_tree
                                .syntax()
                                .parent()
                                .as_ref()
                                // Skip over UseTreeList
                                .and_then(SyntaxNode::parent)
                                .and_then(ast::UseTree::cast)?;
                            let path = use_tree.path()?;
                            let path_segment = path.segment()?;
                            path_segment.name_ref()
                        })
                        .and_then(|name_ref| classify_name_ref(sema, &name_ref))?;

                    Some(NameClass::Definition(name_ref_class.definition(sema.db)))
                } else {
                    let extern_crate = it.syntax().parent().and_then(ast::ExternCrate::cast)?;
                    let resolved = sema.resolve_extern_crate(&extern_crate)?;
                    Some(NameClass::ExternCrate(resolved))
                }
            },
            ast::IdentPat(it) => {
                let local = sema.to_def(&it)?;

                if let Some(record_field_pat) = it.syntax().parent().and_then(ast::RecordPatField::cast) {
                    if record_field_pat.name_ref().is_none() {
                        if let Some(field) = sema.resolve_record_field_pat(&record_field_pat) {
                            let field = Definition::Field(field);
                            return Some(NameClass::FieldShorthand { local, field });
                        }
                    }
                }

                Some(NameClass::Definition(Definition::Local(local)))
            },
            ast::RecordField(it) => {
                let field: hir::Field = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::Field(field)))
            },
            ast::Module(it) => {
                let def = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Struct(it) => {
                let def: hir::Struct = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Union(it) => {
                let def: hir::Union = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Enum(it) => {
                let def: hir::Enum = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Trait(it) => {
                let def: hir::Trait = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Static(it) => {
                let def: hir::Static = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Variant(it) => {
                let def: hir::EnumVariant = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Fn(it) => {
                let def: hir::Function = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::Const(it) => {
                let def: hir::Const = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::TypeAlias(it) => {
                let def: hir::TypeAlias = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::ModuleDef(def.into())))
            },
            ast::MacroCall(it) => {
                let def = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::Macro(def)))
            },
            ast::TypeParam(it) => {
                let def = sema.to_def(&it)?;
                Some(NameClass::Definition(Definition::TypeParam(def)))
            },
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum NameRefClass {
    ExternCrate(Crate),
    Definition(Definition),
    FieldShorthand { local: Local, field: Definition },
}

impl NameRefClass {
    pub fn definition(self, db: &dyn HirDatabase) -> Definition {
        match self {
            NameRefClass::ExternCrate(krate) => Definition::ModuleDef(krate.root_module(db).into()),
            NameRefClass::Definition(def) => def,
            NameRefClass::FieldShorthand { local, field: _ } => Definition::Local(local),
        }
    }
}

// Note: we don't have unit-tests for this rather important function.
// It is primarily exercised via goto definition tests in `ide`.
pub fn classify_name_ref(
    sema: &Semantics<RootDatabase>,
    name_ref: &ast::NameRef,
) -> Option<NameRefClass> {
    let _p = profile::span("classify_name_ref");

    let parent = name_ref.syntax().parent()?;

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        if let Some(func) = sema.resolve_method_call(&method_call) {
            return Some(NameRefClass::Definition(Definition::ModuleDef(func.into())));
        }
    }

    if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
        if let Some(field) = sema.resolve_field(&field_expr) {
            return Some(NameRefClass::Definition(Definition::Field(field)));
        }
    }

    if let Some(record_field) = ast::RecordExprField::for_field_name(name_ref) {
        if let Some((field, local)) = sema.resolve_record_field(&record_field) {
            let field = Definition::Field(field);
            let res = match local {
                None => NameRefClass::Definition(field),
                Some(local) => NameRefClass::FieldShorthand { field, local },
            };
            return Some(res);
        }
    }

    if let Some(record_field_pat) = ast::RecordPatField::cast(parent.clone()) {
        if let Some(field) = sema.resolve_record_field_pat(&record_field_pat) {
            let field = Definition::Field(field);
            return Some(NameRefClass::Definition(field));
        }
    }

    if ast::AssocTypeArg::cast(parent.clone()).is_some() {
        // `Trait<Assoc = Ty>`
        //        ^^^^^
        let path = name_ref.syntax().ancestors().find_map(ast::Path::cast)?;
        let resolved = sema.resolve_path(&path)?;
        if let PathResolution::Def(ModuleDef::Trait(tr)) = resolved {
            if let Some(ty) = tr
                .items(sema.db)
                .iter()
                .filter_map(|assoc| match assoc {
                    hir::AssocItem::TypeAlias(it) => Some(*it),
                    _ => None,
                })
                .find(|alias| alias.name(sema.db).to_string() == **name_ref.text())
            {
                return Some(NameRefClass::Definition(Definition::ModuleDef(
                    ModuleDef::TypeAlias(ty),
                )));
            }
        }
    }

    if let Some(macro_call) = parent.ancestors().find_map(ast::MacroCall::cast) {
        if let Some(path) = macro_call.path() {
            if path.qualifier().is_none() {
                // Only use this to resolve single-segment macro calls like `foo!()`. Multi-segment
                // paths are handled below (allowing `log<|>::info!` to resolve to the log crate).
                if let Some(macro_def) = sema.resolve_macro_call(&macro_call) {
                    return Some(NameRefClass::Definition(Definition::Macro(macro_def)));
                }
            }
        }
    }

    if let Some(path) = name_ref.syntax().ancestors().find_map(ast::Path::cast) {
        if let Some(resolved) = sema.resolve_path(&path) {
            return Some(NameRefClass::Definition(resolved.into()));
        }
    }

    let extern_crate = ast::ExternCrate::cast(parent)?;
    let resolved = sema.resolve_extern_crate(&extern_crate)?;
    Some(NameRefClass::ExternCrate(resolved))
}

impl From<PathResolution> for Definition {
    fn from(path_resolution: PathResolution) -> Self {
        match path_resolution {
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
        }
    }
}
