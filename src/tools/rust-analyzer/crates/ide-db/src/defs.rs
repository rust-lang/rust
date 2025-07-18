//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use crate::RootDatabase;
use crate::documentation::{DocsRangeMap, Documentation, HasDocs};
use crate::famous_defs::FamousDefs;
use arrayvec::ArrayVec;
use either::Either;
use hir::{
    Adt, AsAssocItem, AsExternAssocItem, AssocItem, AttributeTemplate, BuiltinAttr, BuiltinType,
    Const, Crate, DefWithBody, DeriveHelper, DisplayTarget, DocLinkDef, ExternAssocItem,
    ExternCrateDecl, Field, Function, GenericDef, GenericParam, GenericSubstitution, HasContainer,
    HasVisibility, HirDisplay, Impl, InlineAsmOperand, ItemContainer, Label, Local, Macro, Module,
    ModuleDef, Name, PathResolution, Semantics, Static, StaticLifetime, Struct, ToolModule, Trait,
    TraitAlias, TupleField, TypeAlias, Variant, VariantDef, Visibility,
};
use span::Edition;
use stdx::{format_to, impl_from};
use syntax::{
    SyntaxKind, SyntaxNode, SyntaxToken, TextSize,
    ast::{self, AstNode},
    match_ast,
};

// FIXME: a more precise name would probably be `Symbol`?
#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum Definition {
    Macro(Macro),
    Field(Field),
    TupleField(TupleField),
    Module(Module),
    Crate(Crate),
    Function(Function),
    Adt(Adt),
    Variant(Variant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    TraitAlias(TraitAlias),
    TypeAlias(TypeAlias),
    SelfType(Impl),
    GenericParam(GenericParam),
    Local(Local),
    Label(Label),
    DeriveHelper(DeriveHelper),
    BuiltinType(BuiltinType),
    BuiltinLifetime(StaticLifetime),
    BuiltinAttr(BuiltinAttr),
    ToolModule(ToolModule),
    ExternCrateDecl(ExternCrateDecl),
    InlineAsmRegOrRegClass(()),
    InlineAsmOperand(InlineAsmOperand),
}

impl Definition {
    pub fn canonical_module_path(&self, db: &RootDatabase) -> Option<impl Iterator<Item = Module>> {
        self.module(db).map(|it| it.path_to_root(db).into_iter().rev())
    }

    pub fn krate(&self, db: &RootDatabase) -> Option<Crate> {
        Some(match self {
            Definition::Module(m) => m.krate(),
            &Definition::Crate(it) => it,
            _ => self.module(db)?.krate(),
        })
    }

    /// Returns the module this definition resides in.
    ///
    /// As such, for modules themselves this will return the parent module.
    pub fn module(&self, db: &RootDatabase) -> Option<Module> {
        let module = match self {
            Definition::Macro(it) => it.module(db),
            Definition::Module(it) => it.parent(db)?,
            Definition::Crate(_) => return None,
            Definition::Field(it) => it.parent_def(db).module(db),
            Definition::Function(it) => it.module(db),
            Definition::Adt(it) => it.module(db),
            Definition::Const(it) => it.module(db),
            Definition::Static(it) => it.module(db),
            Definition::Trait(it) => it.module(db),
            Definition::TraitAlias(it) => it.module(db),
            Definition::TypeAlias(it) => it.module(db),
            Definition::Variant(it) => it.module(db),
            Definition::SelfType(it) => it.module(db),
            Definition::Local(it) => it.module(db),
            Definition::GenericParam(it) => it.module(db),
            Definition::Label(it) => it.module(db),
            Definition::ExternCrateDecl(it) => it.module(db),
            Definition::DeriveHelper(it) => it.derive().module(db),
            Definition::InlineAsmOperand(it) => it.parent(db).module(db),
            Definition::ToolModule(t) => t.krate().root_module(),
            Definition::BuiltinAttr(_)
            | Definition::BuiltinType(_)
            | Definition::BuiltinLifetime(_)
            | Definition::TupleField(_)
            | Definition::InlineAsmRegOrRegClass(_) => return None,
        };
        Some(module)
    }

    pub fn enclosing_definition(&self, db: &RootDatabase) -> Option<Definition> {
        fn container_to_definition(container: ItemContainer) -> Option<Definition> {
            match container {
                ItemContainer::Trait(it) => Some(it.into()),
                ItemContainer::Impl(it) => Some(it.into()),
                ItemContainer::Module(it) => Some(it.into()),
                ItemContainer::ExternBlock(_) | ItemContainer::Crate(_) => None,
            }
        }
        match self {
            Definition::Macro(it) => Some(it.module(db).into()),
            Definition::Module(it) => it.parent(db).map(Definition::Module),
            Definition::Crate(_) => None,
            Definition::Field(it) => Some(it.parent_def(db).into()),
            Definition::Function(it) => container_to_definition(it.container(db)),
            Definition::Adt(it) => Some(it.module(db).into()),
            Definition::Const(it) => container_to_definition(it.container(db)),
            Definition::Static(it) => container_to_definition(it.container(db)),
            Definition::Trait(it) => container_to_definition(it.container(db)),
            Definition::TraitAlias(it) => container_to_definition(it.container(db)),
            Definition::TypeAlias(it) => container_to_definition(it.container(db)),
            Definition::Variant(it) => Some(Adt::Enum(it.parent_enum(db)).into()),
            Definition::SelfType(it) => Some(it.module(db).into()),
            Definition::Local(it) => it.parent(db).try_into().ok(),
            Definition::GenericParam(it) => Some(it.parent().into()),
            Definition::Label(it) => it.parent(db).try_into().ok(),
            Definition::ExternCrateDecl(it) => container_to_definition(it.container(db)),
            Definition::DeriveHelper(it) => Some(it.derive().module(db).into()),
            Definition::InlineAsmOperand(it) => it.parent(db).try_into().ok(),
            Definition::BuiltinAttr(_)
            | Definition::BuiltinType(_)
            | Definition::BuiltinLifetime(_)
            | Definition::TupleField(_)
            | Definition::ToolModule(_)
            | Definition::InlineAsmRegOrRegClass(_) => None,
        }
    }

    pub fn visibility(&self, db: &RootDatabase) -> Option<Visibility> {
        let vis = match self {
            Definition::Field(sf) => sf.visibility(db),
            Definition::Module(it) => it.visibility(db),
            Definition::Crate(_) => return None,
            Definition::Function(it) => it.visibility(db),
            Definition::Adt(it) => it.visibility(db),
            Definition::Const(it) => it.visibility(db),
            Definition::Static(it) => it.visibility(db),
            Definition::Trait(it) => it.visibility(db),
            Definition::TraitAlias(it) => it.visibility(db),
            Definition::TypeAlias(it) => it.visibility(db),
            Definition::Variant(it) => it.visibility(db),
            Definition::ExternCrateDecl(it) => it.visibility(db),
            Definition::Macro(it) => it.visibility(db),
            Definition::BuiltinType(_) | Definition::TupleField(_) => Visibility::Public,
            Definition::BuiltinAttr(_)
            | Definition::BuiltinLifetime(_)
            | Definition::ToolModule(_)
            | Definition::SelfType(_)
            | Definition::Local(_)
            | Definition::GenericParam(_)
            | Definition::Label(_)
            | Definition::DeriveHelper(_)
            | Definition::InlineAsmRegOrRegClass(_)
            | Definition::InlineAsmOperand(_) => return None,
        };
        Some(vis)
    }

    pub fn name(&self, db: &RootDatabase) -> Option<Name> {
        let name = match self {
            Definition::Macro(it) => it.name(db),
            Definition::Field(it) => it.name(db),
            Definition::Module(it) => it.name(db)?,
            Definition::Crate(it) => {
                Name::new_symbol_root(it.display_name(db)?.crate_name().symbol().clone())
            }
            Definition::Function(it) => it.name(db),
            Definition::Adt(it) => it.name(db),
            Definition::Variant(it) => it.name(db),
            Definition::Const(it) => it.name(db)?,
            Definition::Static(it) => it.name(db),
            Definition::Trait(it) => it.name(db),
            Definition::TraitAlias(it) => it.name(db),
            Definition::TypeAlias(it) => it.name(db),
            Definition::BuiltinType(it) => it.name(),
            Definition::TupleField(it) => it.name(),
            Definition::SelfType(_) => return None,
            Definition::Local(it) => it.name(db),
            Definition::GenericParam(it) => it.name(db),
            Definition::Label(it) => it.name(db),
            Definition::BuiltinLifetime(it) => it.name(),
            Definition::BuiltinAttr(_) => return None, // FIXME
            Definition::ToolModule(_) => return None,  // FIXME
            Definition::DeriveHelper(it) => it.name(db),
            Definition::ExternCrateDecl(it) => return it.alias_or_name(db),
            Definition::InlineAsmRegOrRegClass(_) => return None,
            Definition::InlineAsmOperand(op) => return op.name(db),
        };
        Some(name)
    }

    pub fn docs(
        &self,
        db: &RootDatabase,
        famous_defs: Option<&FamousDefs<'_, '_>>,
        display_target: DisplayTarget,
    ) -> Option<Documentation> {
        self.docs_with_rangemap(db, famous_defs, display_target).map(|(docs, _)| docs)
    }

    pub fn docs_with_rangemap(
        &self,
        db: &RootDatabase,
        famous_defs: Option<&FamousDefs<'_, '_>>,
        display_target: DisplayTarget,
    ) -> Option<(Documentation, Option<DocsRangeMap>)> {
        let docs = match self {
            Definition::Macro(it) => it.docs_with_rangemap(db),
            Definition::Field(it) => it.docs_with_rangemap(db),
            Definition::Module(it) => it.docs_with_rangemap(db),
            Definition::Crate(it) => it.docs_with_rangemap(db),
            Definition::Function(it) => it.docs_with_rangemap(db),
            Definition::Adt(it) => it.docs_with_rangemap(db),
            Definition::Variant(it) => it.docs_with_rangemap(db),
            Definition::Const(it) => it.docs_with_rangemap(db),
            Definition::Static(it) => it.docs_with_rangemap(db),
            Definition::Trait(it) => it.docs_with_rangemap(db),
            Definition::TraitAlias(it) => it.docs_with_rangemap(db),
            Definition::TypeAlias(it) => {
                it.docs_with_rangemap(db).or_else(|| {
                    // docs are missing, try to fall back to the docs of the aliased item.
                    let adt = it.ty(db).as_adt()?;
                    let (docs, range_map) = adt.docs_with_rangemap(db)?;
                    let header_docs = format!(
                        "*This is the documentation for* `{}`\n\n",
                        adt.display(db, display_target)
                    );
                    let offset = TextSize::new(header_docs.len() as u32);
                    let range_map = range_map.shift_docstring_line_range(offset);
                    let docs = header_docs + docs.as_str();
                    Some((Documentation::new(docs), range_map))
                })
            }
            Definition::BuiltinType(it) => {
                famous_defs.and_then(|fd| {
                    // std exposes prim_{} modules with docstrings on the root to document the builtins
                    let primitive_mod =
                        format!("prim_{}", it.name().display(fd.0.db, display_target.edition));
                    let doc_owner = find_std_module(fd, &primitive_mod, display_target.edition)?;
                    doc_owner.docs_with_rangemap(fd.0.db)
                })
            }
            Definition::BuiltinLifetime(StaticLifetime) => None,
            Definition::Local(_) => None,
            Definition::SelfType(impl_def) => {
                impl_def.self_ty(db).as_adt().map(|adt| adt.docs_with_rangemap(db))?
            }
            Definition::GenericParam(_) => None,
            Definition::Label(_) => None,
            Definition::ExternCrateDecl(it) => it.docs_with_rangemap(db),

            Definition::BuiltinAttr(it) => {
                let name = it.name(db);
                let AttributeTemplate { word, list, name_value_str } = it.template(db)?;
                let mut docs = "Valid forms are:".to_owned();
                if word {
                    format_to!(docs, "\n - #\\[{}]", name.display(db, display_target.edition));
                }
                if let Some(list) = list {
                    format_to!(
                        docs,
                        "\n - #\\[{}({})]",
                        name.display(db, display_target.edition),
                        list
                    );
                }
                if let Some(name_value_str) = name_value_str {
                    format_to!(
                        docs,
                        "\n - #\\[{} = {}]",
                        name.display(db, display_target.edition),
                        name_value_str
                    );
                }

                return Some((Documentation::new(docs.replace('*', "\\*")), None));
            }
            Definition::ToolModule(_) => None,
            Definition::DeriveHelper(_) => None,
            Definition::TupleField(_) => None,
            Definition::InlineAsmRegOrRegClass(_) | Definition::InlineAsmOperand(_) => None,
        };

        docs.or_else(|| {
            // docs are missing, for assoc items of trait impls try to fall back to the docs of the
            // original item of the trait
            let assoc = self.as_assoc_item(db)?;
            let trait_ = assoc.implemented_trait(db)?;
            let name = Some(assoc.name(db)?);
            let item = trait_.items(db).into_iter().find(|it| it.name(db) == name)?;
            item.docs_with_rangemap(db)
        })
        .map(|(docs, range_map)| (docs, Some(range_map)))
    }

    pub fn label(&self, db: &RootDatabase, display_target: DisplayTarget) -> String {
        match *self {
            Definition::Macro(it) => it.display(db, display_target).to_string(),
            Definition::Field(it) => it.display(db, display_target).to_string(),
            Definition::TupleField(it) => it.display(db, display_target).to_string(),
            Definition::Module(it) => it.display(db, display_target).to_string(),
            Definition::Crate(it) => it.display(db, display_target).to_string(),
            Definition::Function(it) => it.display(db, display_target).to_string(),
            Definition::Adt(it) => it.display(db, display_target).to_string(),
            Definition::Variant(it) => it.display(db, display_target).to_string(),
            Definition::Const(it) => it.display(db, display_target).to_string(),
            Definition::Static(it) => it.display(db, display_target).to_string(),
            Definition::Trait(it) => it.display(db, display_target).to_string(),
            Definition::TraitAlias(it) => it.display(db, display_target).to_string(),
            Definition::TypeAlias(it) => it.display(db, display_target).to_string(),
            Definition::BuiltinType(it) => {
                it.name().display(db, display_target.edition).to_string()
            }
            Definition::BuiltinLifetime(it) => {
                it.name().display(db, display_target.edition).to_string()
            }
            Definition::Local(it) => {
                let ty = it.ty(db);
                let ty_display = ty.display_truncated(db, None, display_target);
                let is_mut = if it.is_mut(db) { "mut " } else { "" };
                if it.is_self(db) {
                    format!("{is_mut}self: {ty_display}")
                } else {
                    let name = it.name(db);
                    let let_kw = if it.is_param(db) { "" } else { "let " };
                    format!(
                        "{let_kw}{is_mut}{}: {ty_display}",
                        name.display(db, display_target.edition)
                    )
                }
            }
            Definition::SelfType(impl_def) => {
                let self_ty = &impl_def.self_ty(db);
                match self_ty.as_adt() {
                    Some(it) => it.display(db, display_target).to_string(),
                    None => self_ty.display(db, display_target).to_string(),
                }
            }
            Definition::GenericParam(it) => it.display(db, display_target).to_string(),
            Definition::Label(it) => it.name(db).display(db, display_target.edition).to_string(),
            Definition::ExternCrateDecl(it) => it.display(db, display_target).to_string(),
            Definition::BuiltinAttr(it) => {
                format!("#[{}]", it.name(db).display(db, display_target.edition))
            }
            Definition::ToolModule(it) => {
                it.name(db).display(db, display_target.edition).to_string()
            }
            Definition::DeriveHelper(it) => {
                format!("derive_helper {}", it.name(db).display(db, display_target.edition))
            }
            // FIXME
            Definition::InlineAsmRegOrRegClass(_) => "inline_asm_reg_or_reg_class".to_owned(),
            Definition::InlineAsmOperand(_) => "inline_asm_reg_operand".to_owned(),
        }
    }
}

fn find_std_module(
    famous_defs: &FamousDefs<'_, '_>,
    name: &str,
    edition: Edition,
) -> Option<hir::Module> {
    let db = famous_defs.0.db;
    let std_crate = famous_defs.std()?;
    let std_root_module = std_crate.root_module();
    std_root_module.children(db).find(|module| {
        module.name(db).is_some_and(|module| module.display(db, edition).to_string() == name)
    })
}

// FIXME: IdentClass as a name no longer fits
#[derive(Debug)]
pub enum IdentClass<'db> {
    NameClass(NameClass<'db>),
    NameRefClass(NameRefClass<'db>),
    Operator(OperatorClass),
}

impl<'db> IdentClass<'db> {
    pub fn classify_node(
        sema: &Semantics<'db, RootDatabase>,
        node: &SyntaxNode,
    ) -> Option<IdentClass<'db>> {
        match_ast! {
            match node {
                ast::Name(name) => NameClass::classify(sema, &name).map(IdentClass::NameClass),
                ast::NameRef(name_ref) => NameRefClass::classify(sema, &name_ref).map(IdentClass::NameRefClass),
                ast::Lifetime(lifetime) => {
                    NameClass::classify_lifetime(sema, &lifetime)
                        .map(IdentClass::NameClass)
                        .or_else(|| NameRefClass::classify_lifetime(sema, &lifetime).map(IdentClass::NameRefClass))
                },
                ast::RangePat(range_pat) => OperatorClass::classify_range_pat(sema, &range_pat).map(IdentClass::Operator),
                ast::RangeExpr(range_expr) => OperatorClass::classify_range_expr(sema, &range_expr).map(IdentClass::Operator),
                ast::AwaitExpr(await_expr) => OperatorClass::classify_await(sema, &await_expr).map(IdentClass::Operator),
                ast::BinExpr(bin_expr) => OperatorClass::classify_bin(sema, &bin_expr).map(IdentClass::Operator),
                ast::IndexExpr(index_expr) => OperatorClass::classify_index(sema, &index_expr).map(IdentClass::Operator),
                ast::PrefixExpr(prefix_expr) => OperatorClass::classify_prefix(sema, &prefix_expr).map(IdentClass::Operator),
                ast::TryExpr(try_expr) => OperatorClass::classify_try(sema, &try_expr).map(IdentClass::Operator),
                _ => None,
            }
        }
    }

    pub fn classify_token(
        sema: &Semantics<'db, RootDatabase>,
        token: &SyntaxToken,
    ) -> Option<IdentClass<'db>> {
        let parent = token.parent()?;
        Self::classify_node(sema, &parent)
    }

    pub fn classify_lifetime(
        sema: &Semantics<'db, RootDatabase>,
        lifetime: &ast::Lifetime,
    ) -> Option<IdentClass<'db>> {
        NameRefClass::classify_lifetime(sema, lifetime)
            .map(IdentClass::NameRefClass)
            .or_else(|| NameClass::classify_lifetime(sema, lifetime).map(IdentClass::NameClass))
    }

    pub fn definitions(self) -> ArrayVec<(Definition, Option<GenericSubstitution<'db>>), 2> {
        let mut res = ArrayVec::new();
        match self {
            IdentClass::NameClass(NameClass::Definition(it) | NameClass::ConstReference(it)) => {
                res.push((it, None))
            }
            IdentClass::NameClass(NameClass::PatFieldShorthand {
                local_def,
                field_ref,
                adt_subst,
            }) => {
                res.push((Definition::Local(local_def), None));
                res.push((Definition::Field(field_ref), Some(adt_subst)));
            }
            IdentClass::NameRefClass(NameRefClass::Definition(it, subst)) => res.push((it, subst)),
            IdentClass::NameRefClass(NameRefClass::FieldShorthand {
                local_ref,
                field_ref,
                adt_subst,
            }) => {
                res.push((Definition::Local(local_ref), None));
                res.push((Definition::Field(field_ref), Some(adt_subst)));
            }
            IdentClass::NameRefClass(NameRefClass::ExternCrateShorthand { decl, krate }) => {
                res.push((Definition::ExternCrateDecl(decl), None));
                res.push((Definition::Crate(krate), None));
            }
            IdentClass::Operator(
                OperatorClass::Await(func)
                | OperatorClass::Prefix(func)
                | OperatorClass::Bin(func)
                | OperatorClass::Index(func)
                | OperatorClass::Try(func),
            ) => res.push((Definition::Function(func), None)),
            IdentClass::Operator(OperatorClass::Range(struct0)) => {
                res.push((Definition::Adt(Adt::Struct(struct0)), None))
            }
        }
        res
    }

    pub fn definitions_no_ops(self) -> ArrayVec<Definition, 2> {
        let mut res = ArrayVec::new();
        match self {
            IdentClass::NameClass(NameClass::Definition(it) | NameClass::ConstReference(it)) => {
                res.push(it)
            }
            IdentClass::NameClass(NameClass::PatFieldShorthand {
                local_def,
                field_ref,
                adt_subst: _,
            }) => {
                res.push(Definition::Local(local_def));
                res.push(Definition::Field(field_ref));
            }
            IdentClass::NameRefClass(NameRefClass::Definition(it, _)) => res.push(it),
            IdentClass::NameRefClass(NameRefClass::FieldShorthand {
                local_ref,
                field_ref,
                adt_subst: _,
            }) => {
                res.push(Definition::Local(local_ref));
                res.push(Definition::Field(field_ref));
            }
            IdentClass::NameRefClass(NameRefClass::ExternCrateShorthand { decl, krate }) => {
                res.push(Definition::ExternCrateDecl(decl));
                res.push(Definition::Crate(krate));
            }
            IdentClass::Operator(_) => (),
        }
        res
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
pub enum NameClass<'db> {
    Definition(Definition),
    /// `None` in `if let None = Some(82) {}`.
    /// Syntactically, it is a name, but semantically it is a reference.
    ConstReference(Definition),
    /// `field` in `if let Foo { field } = foo`. Here, `ast::Name` both introduces
    /// a definition into a local scope, and refers to an existing definition.
    PatFieldShorthand {
        local_def: Local,
        field_ref: Field,
        adt_subst: GenericSubstitution<'db>,
    },
}

impl<'db> NameClass<'db> {
    /// `Definition` defined by this name.
    pub fn defined(self) -> Option<Definition> {
        let res = match self {
            NameClass::Definition(it) => it,
            NameClass::ConstReference(_) => return None,
            NameClass::PatFieldShorthand { local_def, field_ref: _, adt_subst: _ } => {
                Definition::Local(local_def)
            }
        };
        Some(res)
    }

    pub fn classify(
        sema: &Semantics<'db, RootDatabase>,
        name: &ast::Name,
    ) -> Option<NameClass<'db>> {
        let _p = tracing::info_span!("NameClass::classify").entered();

        let parent = name.syntax().parent()?;
        let definition = match_ast! {
            match parent {
                ast::Item(it) => classify_item(sema, it)?,
                ast::IdentPat(it) => return classify_ident_pat(sema, it),
                ast::Rename(it) => classify_rename(sema, it)?,
                ast::SelfParam(it) => Definition::Local(sema.to_def(&it)?),
                ast::RecordField(it) => Definition::Field(sema.to_def(&it)?),
                ast::Variant(it) => Definition::Variant(sema.to_def(&it)?),
                ast::TypeParam(it) => Definition::GenericParam(sema.to_def(&it)?.into()),
                ast::ConstParam(it) => Definition::GenericParam(sema.to_def(&it)?.into()),
                ast::AsmOperandNamed(it) => Definition::InlineAsmOperand(sema.to_def(&it)?),
                _ => return None,
            }
        };
        return Some(NameClass::Definition(definition));

        fn classify_item(
            sema: &Semantics<'_, RootDatabase>,
            item: ast::Item,
        ) -> Option<Definition> {
            let definition = match item {
                ast::Item::MacroRules(it) => {
                    Definition::Macro(sema.to_def(&ast::Macro::MacroRules(it))?)
                }
                ast::Item::MacroDef(it) => {
                    Definition::Macro(sema.to_def(&ast::Macro::MacroDef(it))?)
                }
                ast::Item::Const(it) => Definition::Const(sema.to_def(&it)?),
                ast::Item::Fn(it) => {
                    let def = sema.to_def(&it)?;
                    def.as_proc_macro(sema.db)
                        .map(Definition::Macro)
                        .unwrap_or(Definition::Function(def))
                }
                ast::Item::Module(it) => Definition::Module(sema.to_def(&it)?),
                ast::Item::Static(it) => Definition::Static(sema.to_def(&it)?),
                ast::Item::Trait(it) => Definition::Trait(sema.to_def(&it)?),
                ast::Item::TraitAlias(it) => Definition::TraitAlias(sema.to_def(&it)?),
                ast::Item::TypeAlias(it) => Definition::TypeAlias(sema.to_def(&it)?),
                ast::Item::Enum(it) => Definition::Adt(hir::Adt::Enum(sema.to_def(&it)?)),
                ast::Item::Struct(it) => Definition::Adt(hir::Adt::Struct(sema.to_def(&it)?)),
                ast::Item::Union(it) => Definition::Adt(hir::Adt::Union(sema.to_def(&it)?)),
                ast::Item::ExternCrate(it) => Definition::ExternCrateDecl(sema.to_def(&it)?),
                _ => return None,
            };
            Some(definition)
        }

        fn classify_ident_pat<'db>(
            sema: &Semantics<'db, RootDatabase>,
            ident_pat: ast::IdentPat,
        ) -> Option<NameClass<'db>> {
            if let Some(def) = sema.resolve_bind_pat_to_const(&ident_pat) {
                return Some(NameClass::ConstReference(Definition::from(def)));
            }

            let local = sema.to_def(&ident_pat)?;
            let pat_parent = ident_pat.syntax().parent();
            if let Some(record_pat_field) = pat_parent.and_then(ast::RecordPatField::cast) {
                if record_pat_field.name_ref().is_none() {
                    if let Some((field, _, adt_subst)) =
                        sema.resolve_record_pat_field_with_subst(&record_pat_field)
                    {
                        return Some(NameClass::PatFieldShorthand {
                            local_def: local,
                            field_ref: field,
                            adt_subst,
                        });
                    }
                }
            }
            Some(NameClass::Definition(Definition::Local(local)))
        }

        fn classify_rename(
            sema: &Semantics<'_, RootDatabase>,
            rename: ast::Rename,
        ) -> Option<Definition> {
            if let Some(use_tree) = rename.syntax().parent().and_then(ast::UseTree::cast) {
                let path = use_tree.path()?;
                sema.resolve_path(&path).map(Definition::from)
            } else {
                sema.to_def(&rename.syntax().parent().and_then(ast::ExternCrate::cast)?)
                    .map(Definition::ExternCrateDecl)
            }
        }
    }

    pub fn classify_lifetime(
        sema: &Semantics<'db, RootDatabase>,
        lifetime: &ast::Lifetime,
    ) -> Option<NameClass<'db>> {
        let _p = tracing::info_span!("NameClass::classify_lifetime", ?lifetime).entered();
        let parent = lifetime.syntax().parent()?;

        if let Some(it) = ast::LifetimeParam::cast(parent.clone()) {
            sema.to_def(&it).map(Into::into).map(Definition::GenericParam)
        } else if let Some(it) = ast::Label::cast(parent) {
            sema.to_def(&it).map(Definition::Label)
        } else {
            None
        }
        .map(NameClass::Definition)
    }
}

#[derive(Debug)]
pub enum OperatorClass {
    Range(Struct),
    Await(Function),
    Prefix(Function),
    Index(Function),
    Try(Function),
    Bin(Function),
}

impl OperatorClass {
    pub fn classify_range_pat(
        sema: &Semantics<'_, RootDatabase>,
        range_pat: &ast::RangePat,
    ) -> Option<OperatorClass> {
        sema.resolve_range_pat(range_pat).map(OperatorClass::Range)
    }

    pub fn classify_range_expr(
        sema: &Semantics<'_, RootDatabase>,
        range_expr: &ast::RangeExpr,
    ) -> Option<OperatorClass> {
        sema.resolve_range_expr(range_expr).map(OperatorClass::Range)
    }

    pub fn classify_await(
        sema: &Semantics<'_, RootDatabase>,
        await_expr: &ast::AwaitExpr,
    ) -> Option<OperatorClass> {
        sema.resolve_await_to_poll(await_expr).map(OperatorClass::Await)
    }

    pub fn classify_prefix(
        sema: &Semantics<'_, RootDatabase>,
        prefix_expr: &ast::PrefixExpr,
    ) -> Option<OperatorClass> {
        sema.resolve_prefix_expr(prefix_expr).map(OperatorClass::Prefix)
    }

    pub fn classify_try(
        sema: &Semantics<'_, RootDatabase>,
        try_expr: &ast::TryExpr,
    ) -> Option<OperatorClass> {
        sema.resolve_try_expr(try_expr).map(OperatorClass::Try)
    }

    pub fn classify_index(
        sema: &Semantics<'_, RootDatabase>,
        index_expr: &ast::IndexExpr,
    ) -> Option<OperatorClass> {
        sema.resolve_index_expr(index_expr).map(OperatorClass::Index)
    }

    pub fn classify_bin(
        sema: &Semantics<'_, RootDatabase>,
        bin_expr: &ast::BinExpr,
    ) -> Option<OperatorClass> {
        sema.resolve_bin_expr(bin_expr).map(OperatorClass::Bin)
    }
}

/// This is similar to [`NameClass`], but works for [`ast::NameRef`] rather than
/// for [`ast::Name`]. Similarly, what looks like a reference in syntax is a
/// reference most of the time, but there are a couple of annoying exceptions.
///
/// A model special case is field shorthand syntax, which uses a single
/// reference to point to two different defs.
#[derive(Debug)]
pub enum NameRefClass<'db> {
    Definition(Definition, Option<GenericSubstitution<'db>>),
    FieldShorthand {
        local_ref: Local,
        field_ref: Field,
        adt_subst: GenericSubstitution<'db>,
    },
    /// The specific situation where we have an extern crate decl without a rename
    /// Here we have both a declaration and a reference.
    /// ```rs
    /// extern crate foo;
    /// ```
    ExternCrateShorthand {
        decl: ExternCrateDecl,
        krate: Crate,
    },
}

impl<'db> NameRefClass<'db> {
    // Note: we don't have unit-tests for this rather important function.
    // It is primarily exercised via goto definition tests in `ide`.
    pub fn classify(
        sema: &Semantics<'db, RootDatabase>,
        name_ref: &ast::NameRef,
    ) -> Option<NameRefClass<'db>> {
        let _p = tracing::info_span!("NameRefClass::classify", ?name_ref).entered();

        let parent = name_ref.syntax().parent()?;

        if let Some(record_field) = ast::RecordExprField::for_field_name(name_ref) {
            if let Some((field, local, _, adt_subst)) =
                sema.resolve_record_field_with_substitution(&record_field)
            {
                let res = match local {
                    None => NameRefClass::Definition(Definition::Field(field), Some(adt_subst)),
                    Some(local) => NameRefClass::FieldShorthand {
                        field_ref: field,
                        local_ref: local,
                        adt_subst,
                    },
                };
                return Some(res);
            }
        }

        if let Some(path) = ast::PathSegment::cast(parent.clone()).map(|it| it.parent_path()) {
            if path.parent_path().is_none() {
                if let Some(macro_call) = path.syntax().parent().and_then(ast::MacroCall::cast) {
                    // Only use this to resolve to macro calls for last segments as qualifiers resolve
                    // to modules below.
                    if let Some(macro_def) = sema.resolve_macro_call(&macro_call) {
                        return Some(NameRefClass::Definition(Definition::Macro(macro_def), None));
                    }
                }
            }
            return sema
                .resolve_path_with_subst(&path)
                .map(|(res, subst)| NameRefClass::Definition(res.into(), subst));
        }

        match_ast! {
            match parent {
                ast::MethodCallExpr(method_call) => {
                    sema.resolve_method_call_fallback(&method_call)
                        .map(|(def, subst)| {
                            match def {
                                Either::Left(def) => NameRefClass::Definition(def.into(), subst),
                                Either::Right(def) => NameRefClass::Definition(def.into(), subst),
                            }
                        })
                },
                ast::FieldExpr(field_expr) => {
                    sema.resolve_field_fallback(&field_expr)
                        .map(|(def, subst)| {
                            match def {
                                Either::Left(Either::Left(def)) => NameRefClass::Definition(def.into(), subst),
                                Either::Left(Either::Right(def)) => NameRefClass::Definition(Definition::TupleField(def), subst),
                                Either::Right(def) => NameRefClass::Definition(def.into(), subst),
                            }
                        })
                },
                ast::RecordPatField(record_pat_field) => {
                    sema.resolve_record_pat_field_with_subst(&record_pat_field)
                        .map(|(field, _, subst)| NameRefClass::Definition(Definition::Field(field), Some(subst)))
                },
                ast::RecordExprField(record_expr_field) => {
                    sema.resolve_record_field_with_substitution(&record_expr_field)
                        .map(|(field, _, _, subst)| NameRefClass::Definition(Definition::Field(field), Some(subst)))
                },
                ast::AssocTypeArg(_) => {
                    // `Trait<Assoc = Ty>`
                    //        ^^^^^
                    let containing_path = name_ref.syntax().ancestors().find_map(ast::Path::cast)?;
                    let resolved = sema.resolve_path(&containing_path)?;
                    if let PathResolution::Def(ModuleDef::Trait(tr)) = resolved {
                        if let Some(ty) = tr
                            .items_with_supertraits(sema.db)
                            .iter()
                            .filter_map(|&assoc| match assoc {
                                hir::AssocItem::TypeAlias(it) => Some(it),
                                _ => None,
                            })
                            .find(|alias| alias.name(sema.db).as_str() == name_ref.text().trim_start_matches("r#"))
                        {
                            // No substitution, this can only occur in type position.
                            return Some(NameRefClass::Definition(Definition::TypeAlias(ty), None));
                        }
                    }
                    None
                },
                ast::UseBoundGenericArgs(_) => {
                    // No substitution, this can only occur in type position.
                    sema.resolve_use_type_arg(name_ref)
                        .map(GenericParam::TypeParam)
                        .map(Definition::GenericParam)
                        .map(|it| NameRefClass::Definition(it, None))
                },
                ast::ExternCrate(extern_crate_ast) => {
                    let extern_crate = sema.to_def(&extern_crate_ast)?;
                    let krate = extern_crate.resolved_crate(sema.db)?;
                    Some(if extern_crate_ast.rename().is_some() {
                        NameRefClass::Definition(Definition::Crate(krate), None)
                    } else {
                        NameRefClass::ExternCrateShorthand { krate, decl: extern_crate }
                    })
                },
                ast::AsmRegSpec(_) => {
                    Some(NameRefClass::Definition(Definition::InlineAsmRegOrRegClass(()), None))
                },
                ast::OffsetOfExpr(_) => {
                    let (def, subst) = sema.resolve_offset_of_field(name_ref)?;
                    let def = match def {
                        Either::Left(variant) => Definition::Variant(variant),
                        Either::Right(field) => Definition::Field(field),
                    };
                    Some(NameRefClass::Definition(def, Some(subst)))
                },
                _ => None
            }
        }
    }

    pub fn classify_lifetime(
        sema: &Semantics<'db, RootDatabase>,
        lifetime: &ast::Lifetime,
    ) -> Option<NameRefClass<'db>> {
        let _p = tracing::info_span!("NameRefClass::classify_lifetime", ?lifetime).entered();
        if lifetime.text() == "'static" {
            return Some(NameRefClass::Definition(
                Definition::BuiltinLifetime(StaticLifetime),
                None,
            ));
        }
        let parent = lifetime.syntax().parent()?;
        match parent.kind() {
            SyntaxKind::BREAK_EXPR | SyntaxKind::CONTINUE_EXPR => sema
                .resolve_label(lifetime)
                .map(Definition::Label)
                .map(|it| NameRefClass::Definition(it, None)),
            SyntaxKind::LIFETIME_ARG
            | SyntaxKind::USE_BOUND_GENERIC_ARGS
            | SyntaxKind::SELF_PARAM
            | SyntaxKind::TYPE_BOUND
            | SyntaxKind::WHERE_PRED
            | SyntaxKind::REF_TYPE => sema
                .resolve_lifetime_param(lifetime)
                .map(GenericParam::LifetimeParam)
                .map(Definition::GenericParam)
                .map(|it| NameRefClass::Definition(it, None)),
            _ => None,
        }
    }
}

impl_from!(
    Field, Module, Function, Adt, Variant, Const, Static, Trait, TraitAlias, TypeAlias, BuiltinType, Local,
    GenericParam, Label, Macro, ExternCrateDecl
    for Definition
);

impl From<Impl> for Definition {
    fn from(impl_: Impl) -> Self {
        Definition::SelfType(impl_)
    }
}

impl From<InlineAsmOperand> for Definition {
    fn from(value: InlineAsmOperand) -> Self {
        Definition::InlineAsmOperand(value)
    }
}

impl From<Either<PathResolution, InlineAsmOperand>> for Definition {
    fn from(value: Either<PathResolution, InlineAsmOperand>) -> Self {
        value.either(Definition::from, Definition::from)
    }
}

impl AsAssocItem for Definition {
    fn as_assoc_item(self, db: &dyn hir::db::HirDatabase) -> Option<AssocItem> {
        match self {
            Definition::Function(it) => it.as_assoc_item(db),
            Definition::Const(it) => it.as_assoc_item(db),
            Definition::TypeAlias(it) => it.as_assoc_item(db),
            _ => None,
        }
    }
}

impl AsExternAssocItem for Definition {
    fn as_extern_assoc_item(self, db: &dyn hir::db::HirDatabase) -> Option<ExternAssocItem> {
        match self {
            Definition::Function(it) => it.as_extern_assoc_item(db),
            Definition::Static(it) => it.as_extern_assoc_item(db),
            Definition::TypeAlias(it) => it.as_extern_assoc_item(db),
            _ => None,
        }
    }
}

impl From<AssocItem> for Definition {
    fn from(assoc_item: AssocItem) -> Self {
        match assoc_item {
            AssocItem::Function(it) => Definition::Function(it),
            AssocItem::Const(it) => Definition::Const(it),
            AssocItem::TypeAlias(it) => Definition::TypeAlias(it),
        }
    }
}

impl From<PathResolution> for Definition {
    fn from(path_resolution: PathResolution) -> Self {
        match path_resolution {
            PathResolution::Def(def) => def.into(),
            PathResolution::Local(local) => Definition::Local(local),
            PathResolution::TypeParam(par) => Definition::GenericParam(par.into()),
            PathResolution::ConstParam(par) => Definition::GenericParam(par.into()),
            PathResolution::SelfType(impl_def) => Definition::SelfType(impl_def),
            PathResolution::BuiltinAttr(attr) => Definition::BuiltinAttr(attr),
            PathResolution::ToolModule(tool) => Definition::ToolModule(tool),
            PathResolution::DeriveHelper(helper) => Definition::DeriveHelper(helper),
        }
    }
}

impl From<ModuleDef> for Definition {
    fn from(def: ModuleDef) -> Self {
        match def {
            ModuleDef::Module(it) => Definition::Module(it),
            ModuleDef::Function(it) => Definition::Function(it),
            ModuleDef::Adt(it) => Definition::Adt(it),
            ModuleDef::Variant(it) => Definition::Variant(it),
            ModuleDef::Const(it) => Definition::Const(it),
            ModuleDef::Static(it) => Definition::Static(it),
            ModuleDef::Trait(it) => Definition::Trait(it),
            ModuleDef::TraitAlias(it) => Definition::TraitAlias(it),
            ModuleDef::TypeAlias(it) => Definition::TypeAlias(it),
            ModuleDef::Macro(it) => Definition::Macro(it),
            ModuleDef::BuiltinType(it) => Definition::BuiltinType(it),
        }
    }
}

impl From<DocLinkDef> for Definition {
    fn from(def: DocLinkDef) -> Self {
        match def {
            DocLinkDef::ModuleDef(it) => it.into(),
            DocLinkDef::Field(it) => it.into(),
            DocLinkDef::SelfType(it) => it.into(),
        }
    }
}

impl From<VariantDef> for Definition {
    fn from(def: VariantDef) -> Self {
        ModuleDef::from(def).into()
    }
}

impl TryFrom<DefWithBody> for Definition {
    type Error = ();
    fn try_from(def: DefWithBody) -> Result<Self, Self::Error> {
        match def {
            DefWithBody::Function(it) => Ok(it.into()),
            DefWithBody::Static(it) => Ok(it.into()),
            DefWithBody::Const(it) => Ok(it.into()),
            DefWithBody::Variant(it) => Ok(it.into()),
        }
    }
}

impl From<GenericDef> for Definition {
    fn from(def: GenericDef) -> Self {
        match def {
            GenericDef::Function(it) => it.into(),
            GenericDef::Adt(it) => it.into(),
            GenericDef::Trait(it) => it.into(),
            GenericDef::TraitAlias(it) => it.into(),
            GenericDef::TypeAlias(it) => it.into(),
            GenericDef::Impl(it) => it.into(),
            GenericDef::Const(it) => it.into(),
            GenericDef::Static(it) => it.into(),
        }
    }
}

impl TryFrom<Definition> for GenericDef {
    type Error = ();
    fn try_from(def: Definition) -> Result<Self, Self::Error> {
        match def {
            Definition::Function(it) => Ok(it.into()),
            Definition::Adt(it) => Ok(it.into()),
            Definition::Trait(it) => Ok(it.into()),
            Definition::TraitAlias(it) => Ok(it.into()),
            Definition::TypeAlias(it) => Ok(it.into()),
            Definition::SelfType(it) => Ok(it.into()),
            Definition::Const(it) => Ok(it.into()),
            _ => Err(()),
        }
    }
}
