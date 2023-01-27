//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

// FIXME: this badly needs rename/rewrite (matklad, 2020-02-06).

use arrayvec::ArrayVec;
use hir::{
    Adt, AsAssocItem, AssocItem, BuiltinAttr, BuiltinType, Const, Crate, DeriveHelper, Field,
    Function, GenericParam, HasVisibility, Impl, ItemInNs, Label, Local, Macro, Module, ModuleDef,
    Name, PathResolution, Semantics, Static, ToolModule, Trait, TypeAlias, Variant, Visibility,
};
use stdx::impl_from;
use syntax::{
    ast::{self, AstNode},
    match_ast, SyntaxKind, SyntaxNode, SyntaxToken,
};

use crate::RootDatabase;

// FIXME: a more precise name would probably be `Symbol`?
#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum Definition {
    Macro(Macro),
    Field(Field),
    Module(Module),
    Function(Function),
    Adt(Adt),
    Variant(Variant),
    Const(Const),
    Static(Static),
    Trait(Trait),
    TypeAlias(TypeAlias),
    BuiltinType(BuiltinType),
    SelfType(Impl),
    GenericParam(GenericParam),
    Local(Local),
    Label(Label),
    DeriveHelper(DeriveHelper),
    BuiltinAttr(BuiltinAttr),
    ToolModule(ToolModule),
}

impl Definition {
    pub fn canonical_module_path(&self, db: &RootDatabase) -> Option<impl Iterator<Item = Module>> {
        self.module(db).map(|it| it.path_to_root(db).into_iter().rev())
    }

    pub fn krate(&self, db: &RootDatabase) -> Option<Crate> {
        Some(match self {
            Definition::Module(m) => m.krate(),
            _ => self.module(db)?.krate(),
        })
    }

    pub fn module(&self, db: &RootDatabase) -> Option<Module> {
        let module = match self {
            Definition::Macro(it) => it.module(db),
            Definition::Module(it) => it.parent(db)?,
            Definition::Field(it) => it.parent_def(db).module(db),
            Definition::Function(it) => it.module(db),
            Definition::Adt(it) => it.module(db),
            Definition::Const(it) => it.module(db),
            Definition::Static(it) => it.module(db),
            Definition::Trait(it) => it.module(db),
            Definition::TypeAlias(it) => it.module(db),
            Definition::Variant(it) => it.module(db),
            Definition::SelfType(it) => it.module(db),
            Definition::Local(it) => it.module(db),
            Definition::GenericParam(it) => it.module(db),
            Definition::Label(it) => it.module(db),
            Definition::DeriveHelper(it) => it.derive().module(db),
            Definition::BuiltinAttr(_) | Definition::BuiltinType(_) | Definition::ToolModule(_) => {
                return None
            }
        };
        Some(module)
    }

    pub fn visibility(&self, db: &RootDatabase) -> Option<Visibility> {
        let vis = match self {
            Definition::Field(sf) => sf.visibility(db),
            Definition::Module(it) => it.visibility(db),
            Definition::Function(it) => it.visibility(db),
            Definition::Adt(it) => it.visibility(db),
            Definition::Const(it) => it.visibility(db),
            Definition::Static(it) => it.visibility(db),
            Definition::Trait(it) => it.visibility(db),
            Definition::TypeAlias(it) => it.visibility(db),
            Definition::Variant(it) => it.visibility(db),
            Definition::BuiltinType(_) => Visibility::Public,
            Definition::Macro(_) => return None,
            Definition::BuiltinAttr(_)
            | Definition::ToolModule(_)
            | Definition::SelfType(_)
            | Definition::Local(_)
            | Definition::GenericParam(_)
            | Definition::Label(_)
            | Definition::DeriveHelper(_) => return None,
        };
        Some(vis)
    }

    pub fn name(&self, db: &RootDatabase) -> Option<Name> {
        let name = match self {
            Definition::Macro(it) => it.name(db),
            Definition::Field(it) => it.name(db),
            Definition::Module(it) => it.name(db)?,
            Definition::Function(it) => it.name(db),
            Definition::Adt(it) => it.name(db),
            Definition::Variant(it) => it.name(db),
            Definition::Const(it) => it.name(db)?,
            Definition::Static(it) => it.name(db),
            Definition::Trait(it) => it.name(db),
            Definition::TypeAlias(it) => it.name(db),
            Definition::BuiltinType(it) => it.name(),
            Definition::SelfType(_) => return None,
            Definition::Local(it) => it.name(db),
            Definition::GenericParam(it) => it.name(db),
            Definition::Label(it) => it.name(db),
            Definition::BuiltinAttr(_) => return None, // FIXME
            Definition::ToolModule(_) => return None,  // FIXME
            Definition::DeriveHelper(it) => it.name(db),
        };
        Some(name)
    }
}

// FIXME: IdentClass as a name no longer fits
#[derive(Debug)]
pub enum IdentClass {
    NameClass(NameClass),
    NameRefClass(NameRefClass),
    Operator(OperatorClass),
}

impl IdentClass {
    pub fn classify_node(
        sema: &Semantics<'_, RootDatabase>,
        node: &SyntaxNode,
    ) -> Option<IdentClass> {
        match_ast! {
            match node {
                ast::Name(name) => NameClass::classify(sema, &name).map(IdentClass::NameClass),
                ast::NameRef(name_ref) => NameRefClass::classify(sema, &name_ref).map(IdentClass::NameRefClass),
                ast::Lifetime(lifetime) => {
                    NameClass::classify_lifetime(sema, &lifetime)
                        .map(IdentClass::NameClass)
                        .or_else(|| NameRefClass::classify_lifetime(sema, &lifetime).map(IdentClass::NameRefClass))
                },
                ast::AwaitExpr(await_expr) => OperatorClass::classify_await(sema, &await_expr).map(IdentClass::Operator),
                ast::BinExpr(bin_expr) => OperatorClass::classify_bin(sema, &bin_expr).map(IdentClass::Operator),
                ast::IndexExpr(index_expr) => OperatorClass::classify_index(sema, &index_expr).map(IdentClass::Operator),
                ast::PrefixExpr(prefix_expr) => OperatorClass::classify_prefix(sema,&prefix_expr).map(IdentClass::Operator),
                ast::TryExpr(try_expr) => OperatorClass::classify_try(sema,&try_expr).map(IdentClass::Operator),
                _ => None,
            }
        }
    }

    pub fn classify_token(
        sema: &Semantics<'_, RootDatabase>,
        token: &SyntaxToken,
    ) -> Option<IdentClass> {
        let parent = token.parent()?;
        Self::classify_node(sema, &parent)
    }

    pub fn classify_lifetime(
        sema: &Semantics<'_, RootDatabase>,
        lifetime: &ast::Lifetime,
    ) -> Option<IdentClass> {
        NameRefClass::classify_lifetime(sema, lifetime)
            .map(IdentClass::NameRefClass)
            .or_else(|| NameClass::classify_lifetime(sema, lifetime).map(IdentClass::NameClass))
    }

    pub fn definitions(self) -> ArrayVec<Definition, 2> {
        let mut res = ArrayVec::new();
        match self {
            IdentClass::NameClass(NameClass::Definition(it) | NameClass::ConstReference(it)) => {
                res.push(it)
            }
            IdentClass::NameClass(NameClass::PatFieldShorthand { local_def, field_ref }) => {
                res.push(Definition::Local(local_def));
                res.push(Definition::Field(field_ref));
            }
            IdentClass::NameRefClass(NameRefClass::Definition(it)) => res.push(it),
            IdentClass::NameRefClass(NameRefClass::FieldShorthand { local_ref, field_ref }) => {
                res.push(Definition::Local(local_ref));
                res.push(Definition::Field(field_ref));
            }
            IdentClass::Operator(
                OperatorClass::Await(func)
                | OperatorClass::Prefix(func)
                | OperatorClass::Bin(func)
                | OperatorClass::Index(func)
                | OperatorClass::Try(func),
            ) => res.push(Definition::Function(func)),
        }
        res
    }

    pub fn definitions_no_ops(self) -> ArrayVec<Definition, 2> {
        let mut res = ArrayVec::new();
        match self {
            IdentClass::NameClass(NameClass::Definition(it) | NameClass::ConstReference(it)) => {
                res.push(it)
            }
            IdentClass::NameClass(NameClass::PatFieldShorthand { local_def, field_ref }) => {
                res.push(Definition::Local(local_def));
                res.push(Definition::Field(field_ref));
            }
            IdentClass::NameRefClass(NameRefClass::Definition(it)) => res.push(it),
            IdentClass::NameRefClass(NameRefClass::FieldShorthand { local_ref, field_ref }) => {
                res.push(Definition::Local(local_ref));
                res.push(Definition::Field(field_ref));
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

    pub fn classify(sema: &Semantics<'_, RootDatabase>, name: &ast::Name) -> Option<NameClass> {
        let _p = profile::span("classify_name");

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
                ast::Item::TypeAlias(it) => Definition::TypeAlias(sema.to_def(&it)?),
                ast::Item::Enum(it) => Definition::Adt(hir::Adt::Enum(sema.to_def(&it)?)),
                ast::Item::Struct(it) => Definition::Adt(hir::Adt::Struct(sema.to_def(&it)?)),
                ast::Item::Union(it) => Definition::Adt(hir::Adt::Union(sema.to_def(&it)?)),
                _ => return None,
            };
            Some(definition)
        }

        fn classify_ident_pat(
            sema: &Semantics<'_, RootDatabase>,
            ident_pat: ast::IdentPat,
        ) -> Option<NameClass> {
            if let Some(def) = sema.resolve_bind_pat_to_const(&ident_pat) {
                return Some(NameClass::ConstReference(Definition::from(def)));
            }

            let local = sema.to_def(&ident_pat)?;
            let pat_parent = ident_pat.syntax().parent();
            if let Some(record_pat_field) = pat_parent.and_then(ast::RecordPatField::cast) {
                if record_pat_field.name_ref().is_none() {
                    if let Some(field) = sema.resolve_record_pat_field(&record_pat_field) {
                        return Some(NameClass::PatFieldShorthand {
                            local_def: local,
                            field_ref: field,
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
                let extern_crate = rename.syntax().parent().and_then(ast::ExternCrate::cast)?;
                let krate = sema.resolve_extern_crate(&extern_crate)?;
                let root_module = krate.root_module(sema.db);
                Some(Definition::Module(root_module))
            }
        }
    }

    pub fn classify_lifetime(
        sema: &Semantics<'_, RootDatabase>,
        lifetime: &ast::Lifetime,
    ) -> Option<NameClass> {
        let _p = profile::span("classify_lifetime").detail(|| lifetime.to_string());
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
    Await(Function),
    Prefix(Function),
    Index(Function),
    Try(Function),
    Bin(Function),
}

impl OperatorClass {
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
pub enum NameRefClass {
    Definition(Definition),
    FieldShorthand { local_ref: Local, field_ref: Field },
}

impl NameRefClass {
    // Note: we don't have unit-tests for this rather important function.
    // It is primarily exercised via goto definition tests in `ide`.
    pub fn classify(
        sema: &Semantics<'_, RootDatabase>,
        name_ref: &ast::NameRef,
    ) -> Option<NameRefClass> {
        let _p = profile::span("classify_name_ref").detail(|| name_ref.to_string());

        let parent = name_ref.syntax().parent()?;

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

        if let Some(path) = ast::PathSegment::cast(parent.clone()).map(|it| it.parent_path()) {
            if path.parent_path().is_none() {
                if let Some(macro_call) = path.syntax().parent().and_then(ast::MacroCall::cast) {
                    // Only use this to resolve to macro calls for last segments as qualifiers resolve
                    // to modules below.
                    if let Some(macro_def) = sema.resolve_macro_call(&macro_call) {
                        return Some(NameRefClass::Definition(Definition::Macro(macro_def)));
                    }
                }
            }
            return sema.resolve_path(&path).map(Into::into).map(NameRefClass::Definition);
        }

        match_ast! {
            match parent {
                ast::MethodCallExpr(method_call) => {
                    sema.resolve_method_call(&method_call)
                        .map(Definition::Function)
                        .map(NameRefClass::Definition)
                },
                ast::FieldExpr(field_expr) => {
                    sema.resolve_field(&field_expr)
                        .map(Definition::Field)
                        .map(NameRefClass::Definition)
                },
                ast::RecordPatField(record_pat_field) => {
                    sema.resolve_record_pat_field(&record_pat_field)
                        .map(Definition::Field)
                        .map(NameRefClass::Definition)
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
                            .find(|alias| alias.name(sema.db).to_smol_str() == name_ref.text().as_str())
                        {
                            return Some(NameRefClass::Definition(Definition::TypeAlias(ty)));
                        }
                    }
                    None
                },
                ast::ExternCrate(extern_crate) => {
                    let krate = sema.resolve_extern_crate(&extern_crate)?;
                    let root_module = krate.root_module(sema.db);
                    Some(NameRefClass::Definition(Definition::Module(root_module)))
                },
                _ => None
            }
        }
    }

    pub fn classify_lifetime(
        sema: &Semantics<'_, RootDatabase>,
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

impl_from!(
    Field, Module, Function, Adt, Variant, Const, Static, Trait, TypeAlias, BuiltinType, Local,
    GenericParam, Label, Macro
    for Definition
);

impl From<Impl> for Definition {
    fn from(impl_: Impl) -> Self {
        Definition::SelfType(impl_)
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
            ModuleDef::TypeAlias(it) => Definition::TypeAlias(it),
            ModuleDef::Macro(it) => Definition::Macro(it),
            ModuleDef::BuiltinType(it) => Definition::BuiltinType(it),
        }
    }
}

impl From<Definition> for Option<ItemInNs> {
    fn from(def: Definition) -> Self {
        let item = match def {
            Definition::Module(it) => ModuleDef::Module(it),
            Definition::Function(it) => ModuleDef::Function(it),
            Definition::Adt(it) => ModuleDef::Adt(it),
            Definition::Variant(it) => ModuleDef::Variant(it),
            Definition::Const(it) => ModuleDef::Const(it),
            Definition::Static(it) => ModuleDef::Static(it),
            Definition::Trait(it) => ModuleDef::Trait(it),
            Definition::TypeAlias(it) => ModuleDef::TypeAlias(it),
            Definition::BuiltinType(it) => ModuleDef::BuiltinType(it),
            _ => return None,
        };
        Some(ItemInNs::from(item))
    }
}
