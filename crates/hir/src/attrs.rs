//! Attributes & documentation for hir types.

use hir_def::{
    attr::AttrsWithOwner,
    item_scope::ItemInNs,
    path::{ModPath, Path},
    per_ns::Namespace,
    resolver::{HasResolver, Resolver, TypeNs},
    AssocItemId, AttrDefId, ModuleDefId,
};
use hir_expand::{hygiene::Hygiene, name::Name};
use hir_ty::db::HirDatabase;
use syntax::{ast, AstNode};

use crate::{
    Adt, AsAssocItem, AssocItem, BuiltinType, Const, ConstParam, DocLinkDef, Enum, ExternCrateDecl,
    Field, Function, GenericParam, Impl, LifetimeParam, Macro, Module, ModuleDef, Static, Struct,
    Trait, TraitAlias, TypeAlias, TypeParam, Union, Variant, VariantDef,
};

pub trait HasAttrs {
    fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner;
    #[doc(hidden)]
    fn attr_id(self) -> AttrDefId;
}

macro_rules! impl_has_attrs {
    ($(($def:ident, $def_id:ident),)*) => {$(
        impl HasAttrs for $def {
            fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner {
                let def = AttrDefId::$def_id(self.into());
                db.attrs_with_owner(def)
            }
            fn attr_id(self) -> AttrDefId {
                AttrDefId::$def_id(self.into())
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
    (ExternCrateDecl, ExternCrateId),
];

macro_rules! impl_has_attrs_enum {
    ($($variant:ident),* for $enum:ident) => {$(
        impl HasAttrs for $variant {
            fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner {
                $enum::$variant(self).attrs(db)
            }
            fn attr_id(self) -> AttrDefId {
                $enum::$variant(self).attr_id()
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
    fn attr_id(self) -> AttrDefId {
        match self {
            AssocItem::Function(it) => it.attr_id(),
            AssocItem::Const(it) => it.attr_id(),
            AssocItem::TypeAlias(it) => it.attr_id(),
        }
    }
}

/// Resolves the item `link` points to in the scope of `def`.
pub fn resolve_doc_path_on(
    db: &dyn HirDatabase,
    def: impl HasAttrs,
    link: &str,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    // AttrDefId::FieldId(it) => it.parent.resolver(db.upcast()),
    // AttrDefId::EnumVariantId(it) => it.parent.resolver(db.upcast()),

    resolve_doc_path_on_(db, link, def.attr_id(), ns)
}

fn resolve_doc_path_on_(
    db: &dyn HirDatabase,
    link: &str,
    attr_id: AttrDefId,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    let resolver = match attr_id {
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
        AttrDefId::UseId(it) => it.resolver(db.upcast()),
        AttrDefId::MacroId(it) => it.resolver(db.upcast()),
        AttrDefId::ExternCrateId(it) => it.resolver(db.upcast()),
        AttrDefId::GenericParamId(_) => return None,
    };

    let mut modpath = modpath_from_str(db, link)?;

    let resolved = resolver.resolve_module_path_in_items(db.upcast(), &modpath);
    if resolved.is_none() {
        let last_name = modpath.pop_segment()?;
        resolve_assoc_or_field(db, resolver, modpath, last_name, ns)
    } else {
        let def = match ns {
            Some(Namespace::Types) => resolved.take_types(),
            Some(Namespace::Values) => resolved.take_values(),
            Some(Namespace::Macros) => resolved.take_macros().map(ModuleDefId::MacroId),
            None => resolved.iter_items().next().map(|(it, _)| match it {
                ItemInNs::Types(it) => it,
                ItemInNs::Values(it) => it,
                ItemInNs::Macros(it) => ModuleDefId::MacroId(it),
            }),
        };
        Some(DocLinkDef::ModuleDef(def?.into()))
    }
}

fn resolve_assoc_or_field(
    db: &dyn HirDatabase,
    resolver: Resolver,
    path: ModPath,
    name: Name,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    let path = Path::from_known_path_with_no_generic(path);
    // FIXME: This does not handle `Self` on trait definitions, which we should resolve to the
    // trait itself.
    let base_def = resolver.resolve_path_in_type_ns_fully(db.upcast(), &path)?;

    let ty = match base_def {
        TypeNs::SelfType(id) => Impl::from(id).self_ty(db),
        TypeNs::GenericParam(_) => {
            // Even if this generic parameter has some trait bounds, rustdoc doesn't
            // resolve `name` to trait items.
            return None;
        }
        TypeNs::AdtId(id) | TypeNs::AdtSelfType(id) => Adt::from(id).ty(db),
        TypeNs::EnumVariantId(id) => {
            // Enum variants don't have path candidates.
            let variant = Variant::from(id);
            return resolve_field(db, variant.into(), name, ns);
        }
        TypeNs::TypeAliasId(id) => {
            let alias = TypeAlias::from(id);
            if alias.as_assoc_item(db).is_some() {
                // We don't normalize associated type aliases, so we have nothing to
                // resolve `name` to.
                return None;
            }
            alias.ty(db)
        }
        TypeNs::BuiltinType(id) => BuiltinType::from(id).ty(db),
        TypeNs::TraitId(id) => {
            // Doc paths in this context may only resolve to an item of this trait
            // (i.e. no items of its supertraits), so we need to handle them here
            // independently of others.
            return db.trait_data(id).items.iter().find(|it| it.0 == name).map(|(_, assoc_id)| {
                let def = match *assoc_id {
                    AssocItemId::FunctionId(it) => ModuleDef::Function(it.into()),
                    AssocItemId::ConstId(it) => ModuleDef::Const(it.into()),
                    AssocItemId::TypeAliasId(it) => ModuleDef::TypeAlias(it.into()),
                };
                DocLinkDef::ModuleDef(def)
            });
        }
        TypeNs::TraitAliasId(_) => {
            // XXX: Do these get resolved?
            return None;
        }
    };

    // FIXME: Resolve associated items here, e.g. `Option::map`. Note that associated items take
    // precedence over fields.

    let variant_def = match ty.as_adt()? {
        Adt::Struct(it) => it.into(),
        Adt::Union(it) => it.into(),
        Adt::Enum(_) => return None,
    };
    resolve_field(db, variant_def, name, ns)
}

fn resolve_field(
    db: &dyn HirDatabase,
    def: VariantDef,
    name: Name,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    if let Some(Namespace::Types | Namespace::Macros) = ns {
        return None;
    }
    def.fields(db).into_iter().find(|f| f.name(db) == name).map(DocLinkDef::Field)
}

fn modpath_from_str(db: &dyn HirDatabase, link: &str) -> Option<ModPath> {
    // FIXME: this is not how we should get a mod path here.
    let try_get_modpath = |link: &str| {
        let ast_path = ast::SourceFile::parse(&format!("type T = {link};"))
            .syntax_node()
            .descendants()
            .find_map(ast::Path::cast)?;
        if ast_path.syntax().text() != link {
            return None;
        }
        ModPath::from_src(db.upcast(), ast_path, &Hygiene::new_unhygienic())
    };

    let full = try_get_modpath(link);
    if full.is_some() {
        return full;
    }

    // Tuple field names cannot be a part of `ModPath` usually, but rustdoc can
    // resolve doc paths like `TupleStruct::0`.
    // FIXME: Find a better way to handle these.
    let (base, maybe_tuple_field) = link.rsplit_once("::")?;
    let tuple_field = Name::new_tuple_field(maybe_tuple_field.parse().ok()?);
    let mut modpath = try_get_modpath(base)?;
    modpath.push_segment(tuple_field);
    Some(modpath)
}
