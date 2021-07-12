//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use hir::{
    Field, GenericParam, HasVisibility, Impl, Label, Local, MacroDef, Module, ModuleDef, Name,
    PathResolution, Semantics, Visibility,
};
use syntax::{
    ast::{self, AstNode, PathSegmentKind},
    match_ast, SyntaxKind, SyntaxNode,
};

use crate::RootDatabase;

// FIXME: a more precise name would probably be `Symbol`?
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Definition {
    Macro(MacroDef),
    Field(Field),
    ModuleDef(ModuleDef),
    SelfType(Impl),
    Local(Local),
    GenericParam(GenericParam),
    Label(Label),
}

impl Definition {
    pub fn module(&self, db: &RootDatabase) -> Option<Module> {
        match self {
            Definition::Macro(it) => it.module(db),
            Definition::Field(it) => Some(it.parent_def(db).module(db)),
            Definition::ModuleDef(it) => it.module(db),
            Definition::SelfType(it) => Some(it.module(db)),
            Definition::Local(it) => Some(it.module(db)),
            Definition::GenericParam(it) => Some(it.module(db)),
            Definition::Label(it) => Some(it.module(db)),
        }
    }

    pub fn visibility(&self, db: &RootDatabase) -> Option<Visibility> {
        match self {
            Definition::Field(sf) => Some(sf.visibility(db)),
            Definition::ModuleDef(def) => match def {
                ModuleDef::Module(it) => Some(it.visibility(db)),
                ModuleDef::Function(it) => Some(it.visibility(db)),
                ModuleDef::Adt(it) => Some(it.visibility(db)),
                ModuleDef::Const(it) => Some(it.visibility(db)),
                ModuleDef::Static(it) => Some(it.visibility(db)),
                ModuleDef::Trait(it) => Some(it.visibility(db)),
                ModuleDef::TypeAlias(it) => Some(it.visibility(db)),
                // NB: Variants don't have their own visibility, and just inherit
                // one from the parent. Not sure if that's the right thing to do.
                ModuleDef::Variant(it) => Some(it.parent_enum(db).visibility(db)),
                ModuleDef::BuiltinType(_) => None,
            },
            Definition::Macro(_)
            | Definition::SelfType(_)
            | Definition::Local(_)
            | Definition::GenericParam(_)
            | Definition::Label(_) => None,
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
                hir::ModuleDef::Variant(it) => it.name(db),
                hir::ModuleDef::Const(it) => it.name(db)?,
                hir::ModuleDef::Static(it) => it.name(db)?,
                hir::ModuleDef::Trait(it) => it.name(db),
                hir::ModuleDef::TypeAlias(it) => it.name(db),
                hir::ModuleDef::BuiltinType(it) => it.name(),
            },
            Definition::SelfType(_) => return None,
            Definition::Local(it) => it.name(db)?,
            Definition::GenericParam(it) => it.name(db),
            Definition::Label(it) => it.name(db),
        };
        Some(name)
    }
}

/// On a first blush, a single `ast::Name` defines a single definition at some
/// scope. That is, that, by just looking at the syntactical category, we can
/// unambiguously define the semantic category.
///
/// Sadly, that's not 100% true, there are special cases. To make sure that
/// callers handle all the special cases correctly via exhaustive matching, we
/// add a [`NameClass`] enum which lists all of them!
///
/// A model special case is `None` constant in pattern.
#[derive(Debug)]
pub enum NameClass {
    Definition(Definition),
    /// `None` in `if let None = Some(82) {}`.
    /// Syntactically, it is a name, but semantically it is a reference.
    ConstReference(Definition),
    /// `field` in `if let Foo { field } = foo`. Here, `ast::Name` both introduces
    /// a definition into a local scope, and refers to an existing definition.
    PatFieldShorthand {
        local_def: Local,
        field_ref: Field,
    },
}

impl NameClass {
    /// `Definition` defined by this name.
    pub fn defined(self) -> Option<Definition> {
        let res = match self {
            NameClass::Definition(it) => it,
            NameClass::ConstReference(_) => return None,
            NameClass::PatFieldShorthand { local_def, field_ref: _ } => {
                Definition::Local(local_def)
            }
        };
        Some(res)
    }

    pub fn classify(sema: &Semantics<RootDatabase>, name: &ast::Name) -> Option<NameClass> {
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
                            .kind()
                            .and_then(|kind| {
                                match kind {
                                    // The rename might be from a `self` token, so fallback to the name higher
                                    // in the use tree.
                                    PathSegmentKind::SelfKw => {
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
                                    },
                                    PathSegmentKind::Name(name_ref) => Some(name_ref),
                                    _ => return None,
                                }
                            })
                            .and_then(|name_ref| NameRefClass::classify(sema, &name_ref))?;

                        Some(NameClass::Definition(match name_ref_class {
                            NameRefClass::Definition(def) => def,
                            NameRefClass::FieldShorthand { local_ref: _, field_ref } => {
                                Definition::Field(field_ref)
                            }
                        }))
                    } else {
                        let extern_crate = it.syntax().parent().and_then(ast::ExternCrate::cast)?;
                        let krate = sema.resolve_extern_crate(&extern_crate)?;
                        let root_module = krate.root_module(sema.db);
                        Some(NameClass::Definition(Definition::ModuleDef(root_module.into())))
                    }
                },
                ast::IdentPat(it) => {
                    let local = sema.to_def(&it)?;

                    if let Some(record_pat_field) = it.syntax().parent().and_then(ast::RecordPatField::cast) {
                        if record_pat_field.name_ref().is_none() {
                            if let Some(field) = sema.resolve_record_pat_field(&record_pat_field) {
                                return Some(NameClass::PatFieldShorthand { local_def: local, field_ref: field });
                            }
                        }
                    }

                    Some(NameClass::Definition(Definition::Local(local)))
                },
                ast::SelfParam(it) => {
                    let def = sema.to_def(&it)?;
                    Some(NameClass::Definition(Definition::Local(def)))
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
                    let def: hir::Variant = sema.to_def(&it)?;
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
                ast::Macro(it) => {
                    let def = sema.to_def(&it)?;
                    Some(NameClass::Definition(Definition::Macro(def)))
                },
                ast::TypeParam(it) => {
                    let def = sema.to_def(&it)?;
                    Some(NameClass::Definition(Definition::GenericParam(def.into())))
                },
                ast::ConstParam(it) => {
                    let def = sema.to_def(&it)?;
                    Some(NameClass::Definition(Definition::GenericParam(def.into())))
                },
                _ => None,
            }
        }
    }

    pub fn classify_lifetime(
        sema: &Semantics<RootDatabase>,
        lifetime: &ast::Lifetime,
    ) -> Option<NameClass> {
        let _p = profile::span("classify_lifetime").detail(|| lifetime.to_string());
        let parent = lifetime.syntax().parent()?;

        match_ast! {
            match parent {
                ast::LifetimeParam(it) => {
                    let def = sema.to_def(&it)?;
                    Some(NameClass::Definition(Definition::GenericParam(def.into())))
                },
                ast::Label(it) => {
                    let def = sema.to_def(&it)?;
                    Some(NameClass::Definition(Definition::Label(def)))
                },
                _ => None,
            }
        }
    }
}

/// This is similar to [`NameClass`], but works for [`ast::NameRef`] rather than
/// for [`ast::Name`]. Similarly, what looks like a reference in syntax is a
/// reference most of the time, but there are a couple of annoying exceptions.
///
/// A model special case is field shorthand syntax, which uses a single
/// reference to point to two different defs.
#[derive(Debug)]
pub enum NameRefClass {
    Definition(Definition),
    FieldShorthand { local_ref: Local, field_ref: Field },
}

impl NameRefClass {
    // Note: we don't have unit-tests for this rather important function.
    // It is primarily exercised via goto definition tests in `ide`.
    pub fn classify(
        sema: &Semantics<RootDatabase>,
        name_ref: &ast::NameRef,
    ) -> Option<NameRefClass> {
        let _p = profile::span("classify_name_ref").detail(|| name_ref.to_string());

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
            if let Some((field, local, _)) = sema.resolve_record_field(&record_field) {
                let res = match local {
                    None => NameRefClass::Definition(Definition::Field(field)),
                    Some(local) => {
                        NameRefClass::FieldShorthand { field_ref: field, local_ref: local }
                    }
                };
                return Some(res);
            }
        }

        if let Some(record_pat_field) = ast::RecordPatField::cast(parent.clone()) {
            if let Some(field) = sema.resolve_record_pat_field(&record_pat_field) {
                let field = Definition::Field(field);
                return Some(NameRefClass::Definition(field));
            }
        }

        if let Some(assoc_type_arg) = ast::AssocTypeArg::cast(parent.clone()) {
            if assoc_type_arg.name_ref().as_ref() == Some(name_ref) {
                // `Trait<Assoc = Ty>`
                //        ^^^^^
                let path = name_ref.syntax().ancestors().find_map(ast::Path::cast)?;
                let resolved = sema.resolve_path(&path)?;
                if let PathResolution::Def(ModuleDef::Trait(tr)) = resolved {
                    // FIXME: resolve in supertraits
                    if let Some(ty) = tr
                        .items(sema.db)
                        .iter()
                        .filter_map(|assoc| match assoc {
                            hir::AssocItem::TypeAlias(it) => Some(*it),
                            _ => None,
                        })
                        .find(|alias| &alias.name(sema.db).to_string() == &name_ref.text())
                    {
                        return Some(NameRefClass::Definition(Definition::ModuleDef(
                            ModuleDef::TypeAlias(ty),
                        )));
                    }
                }

                return None;
            }
        }

        if let Some(path) = name_ref.syntax().ancestors().find_map(ast::Path::cast) {
            if path.qualifier().is_none() {
                if let Some(macro_call) = path.syntax().parent().and_then(ast::MacroCall::cast) {
                    // Only use this to resolve single-segment macro calls like `foo!()`. Multi-segment
                    // paths are handled below (allowing `log$0::info!` to resolve to the log crate).
                    if let Some(macro_def) = sema.resolve_macro_call(&macro_call) {
                        return Some(NameRefClass::Definition(Definition::Macro(macro_def)));
                    }
                }
            }
            let top_path = path.top_path();
            let is_attribute_path = top_path
                .syntax()
                .ancestors()
                .find_map(ast::Attr::cast)
                .map(|attr| attr.path().as_ref() == Some(&top_path));
            return match is_attribute_path {
                Some(true) => sema.resolve_path(&path).and_then(|resolved| {
                    match resolved {
                        // Don't wanna collide with builtin attributes here like `test` hence guard
                        // so only resolve to modules that aren't the last segment
                        PathResolution::Def(module @ ModuleDef::Module(_)) if path != top_path => {
                            cov_mark::hit!(name_ref_classify_attr_path_qualifier);
                            Some(NameRefClass::Definition(Definition::ModuleDef(module)))
                        }
                        PathResolution::Macro(mac) if mac.kind() == hir::MacroKind::Attr => {
                            Some(NameRefClass::Definition(Definition::Macro(mac)))
                        }
                        _ => None,
                    }
                }),
                Some(false) => None,
                None => sema.resolve_path(&path).map(Into::into).map(NameRefClass::Definition),
            };
        }

        let extern_crate = ast::ExternCrate::cast(parent)?;
        let krate = sema.resolve_extern_crate(&extern_crate)?;
        let root_module = krate.root_module(sema.db);
        Some(NameRefClass::Definition(Definition::ModuleDef(root_module.into())))
    }

    pub fn classify_lifetime(
        sema: &Semantics<RootDatabase>,
        lifetime: &ast::Lifetime,
    ) -> Option<NameRefClass> {
        let _p = profile::span("classify_lifetime_ref").detail(|| lifetime.to_string());
        let parent = lifetime.syntax().parent()?;
        match parent.kind() {
            SyntaxKind::BREAK_EXPR | SyntaxKind::CONTINUE_EXPR => {
                sema.resolve_label(lifetime).map(Definition::Label).map(NameRefClass::Definition)
            }
            SyntaxKind::LIFETIME_ARG
            | SyntaxKind::SELF_PARAM
            | SyntaxKind::TYPE_BOUND
            | SyntaxKind::WHERE_PRED
            | SyntaxKind::REF_TYPE => sema
                .resolve_lifetime_param(lifetime)
                .map(GenericParam::LifetimeParam)
                .map(Definition::GenericParam)
                .map(NameRefClass::Definition),
            // lifetime bounds, as in the 'b in 'a: 'b aren't wrapped in TypeBound nodes so we gotta check
            // if our lifetime is in a LifetimeParam without being the constrained lifetime
            _ if ast::LifetimeParam::cast(parent).and_then(|param| param.lifetime()).as_ref()
                != Some(lifetime) =>
            {
                sema.resolve_lifetime_param(lifetime)
                    .map(GenericParam::LifetimeParam)
                    .map(Definition::GenericParam)
                    .map(NameRefClass::Definition)
            }
            _ => None,
        }
    }
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
            PathResolution::TypeParam(par) => Definition::GenericParam(par.into()),
            PathResolution::Macro(def) => Definition::Macro(def),
            PathResolution::SelfType(impl_def) => Definition::SelfType(impl_def),
            PathResolution::ConstParam(par) => Definition::GenericParam(par.into()),
        }
    }
}
