//! Attributes & documentation for hir types.

use hir_def::{
    attr::{AttrsWithOwner, Documentation},
    item_scope::ItemInNs,
    path::ModPath,
    resolver::HasResolver,
    AttrDefId, GenericParamId, ModuleDefId,
};
use hir_expand::hygiene::Hygiene;
use hir_ty::db::HirDatabase;
use syntax::{ast, AstNode};

use crate::{
    Adt, AssocItem, Const, ConstParam, Enum, Field, Function, GenericParam, Impl, LifetimeParam,
    Macro, Module, ModuleDef, Static, Struct, Trait, TraitAlias, TypeAlias, TypeParam, Union,
    Variant,
};

pub trait HasAttrs {
    fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner;
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation>;
    fn resolve_doc_path(
        self,
        db: &dyn HirDatabase,
        link: &str,
        ns: Option<Namespace>,
    ) -> Option<ModuleDef>;
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Namespace {
    Types,
    Values,
    Macros,
}

macro_rules! impl_has_attrs {
    ($(($def:ident, $def_id:ident),)*) => {$(
        impl HasAttrs for $def {
            fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner {
                let def = AttrDefId::$def_id(self.into());
                db.attrs_with_owner(def)
            }
            fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
                let def = AttrDefId::$def_id(self.into());
                db.attrs(def).docs()
            }
            fn resolve_doc_path(self, db: &dyn HirDatabase, link: &str, ns: Option<Namespace>) -> Option<ModuleDef> {
                let def = AttrDefId::$def_id(self.into());
                resolve_doc_path(db, def, link, ns).map(ModuleDef::from)
            }
        }
    )*};
}

impl_has_attrs![
    (Field, FieldId),
    (Variant, EnumVariantId),
    (Static, StaticId),
    (Const, ConstId),
    (Trait, TraitId),
    (TraitAlias, TraitAliasId),
    (TypeAlias, TypeAliasId),
    (Macro, MacroId),
    (Function, FunctionId),
    (Adt, AdtId),
    (Module, ModuleId),
    (GenericParam, GenericParamId),
    (Impl, ImplId),
];

macro_rules! impl_has_attrs_enum {
    ($($variant:ident),* for $enum:ident) => {$(
        impl HasAttrs for $variant {
            fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner {
                $enum::$variant(self).attrs(db)
            }
            fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
                $enum::$variant(self).docs(db)
            }
            fn resolve_doc_path(self, db: &dyn HirDatabase, link: &str, ns: Option<Namespace>) -> Option<ModuleDef> {
                $enum::$variant(self).resolve_doc_path(db, link, ns)
            }
        }
    )*};
}

impl_has_attrs_enum![Struct, Union, Enum for Adt];
impl_has_attrs_enum![TypeParam, ConstParam, LifetimeParam for GenericParam];

impl HasAttrs for AssocItem {
    fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner {
        match self {
            AssocItem::Function(it) => it.attrs(db),
            AssocItem::Const(it) => it.attrs(db),
            AssocItem::TypeAlias(it) => it.attrs(db),
        }
    }

    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
        match self {
            AssocItem::Function(it) => it.docs(db),
            AssocItem::Const(it) => it.docs(db),
            AssocItem::TypeAlias(it) => it.docs(db),
        }
    }

    fn resolve_doc_path(
        self,
        db: &dyn HirDatabase,
        link: &str,
        ns: Option<Namespace>,
    ) -> Option<ModuleDef> {
        match self {
            AssocItem::Function(it) => it.resolve_doc_path(db, link, ns),
            AssocItem::Const(it) => it.resolve_doc_path(db, link, ns),
            AssocItem::TypeAlias(it) => it.resolve_doc_path(db, link, ns),
        }
    }
}

/// Resolves the item `link` points to in the scope of `def`.
fn resolve_doc_path(
    db: &dyn HirDatabase,
    def: AttrDefId,
    link: &str,
    ns: Option<Namespace>,
) -> Option<ModuleDefId> {
    let resolver = match def {
        AttrDefId::ModuleId(it) => it.resolver(db.upcast()),
        AttrDefId::FieldId(it) => it.parent.resolver(db.upcast()),
        AttrDefId::AdtId(it) => it.resolver(db.upcast()),
        AttrDefId::FunctionId(it) => it.resolver(db.upcast()),
        AttrDefId::EnumVariantId(it) => it.parent.resolver(db.upcast()),
        AttrDefId::StaticId(it) => it.resolver(db.upcast()),
        AttrDefId::ConstId(it) => it.resolver(db.upcast()),
        AttrDefId::TraitId(it) => it.resolver(db.upcast()),
        AttrDefId::TraitAliasId(it) => it.resolver(db.upcast()),
        AttrDefId::TypeAliasId(it) => it.resolver(db.upcast()),
        AttrDefId::ImplId(it) => it.resolver(db.upcast()),
        AttrDefId::ExternBlockId(it) => it.resolver(db.upcast()),
        AttrDefId::MacroId(it) => it.resolver(db.upcast()),
        AttrDefId::GenericParamId(it) => match it {
            GenericParamId::TypeParamId(it) => it.parent(),
            GenericParamId::ConstParamId(it) => it.parent(),
            GenericParamId::LifetimeParamId(it) => it.parent,
        }
        .resolver(db.upcast()),
    };

    let modpath = {
        // FIXME: this is not how we should get a mod path here
        let ast_path = ast::SourceFile::parse(&format!("type T = {link};"))
            .syntax_node()
            .descendants()
            .find_map(ast::Path::cast)?;
        if ast_path.syntax().text() != link {
            return None;
        }
        ModPath::from_src(db.upcast(), ast_path, &Hygiene::new_unhygienic())?
    };

    let resolved = resolver.resolve_module_path_in_items(db.upcast(), &modpath);
    let resolved = if resolved.is_none() {
        resolver.resolve_module_path_in_trait_assoc_items(db.upcast(), &modpath)?
    } else {
        resolved
    };
    match ns {
        Some(Namespace::Types) => resolved.take_types(),
        Some(Namespace::Values) => resolved.take_values(),
        Some(Namespace::Macros) => resolved.take_macros().map(ModuleDefId::MacroId),
        None => resolved.iter_items().next().map(|it| match it {
            ItemInNs::Types(it) => it,
            ItemInNs::Values(it) => it,
            ItemInNs::Macros(it) => ModuleDefId::MacroId(it),
        }),
    }
}
