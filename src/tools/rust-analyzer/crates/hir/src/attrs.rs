//! Attributes & documentation for hir types.

use std::ops::ControlFlow;

use hir_def::{
    AssocItemId, AttrDefId, ModuleDefId,
    attr::AttrsWithOwner,
    expr_store::path::Path,
    item_scope::ItemInNs,
    per_ns::Namespace,
    resolver::{HasResolver, Resolver, TypeNs},
};
use hir_expand::{
    mod_path::{ModPath, PathKind},
    name::Name,
};
use hir_ty::{db::HirDatabase, method_resolution};

use crate::{
    Adt, AsAssocItem, AssocItem, BuiltinType, Const, ConstParam, DocLinkDef, Enum, ExternCrateDecl,
    Field, Function, GenericParam, HasCrate, Impl, LifetimeParam, Macro, Module, ModuleDef, Static,
    Struct, Trait, TraitAlias, Type, TypeAlias, TypeParam, Union, Variant, VariantDef,
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
                AttrsWithOwner::new(db, def)
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

impl HasAttrs for crate::Crate {
    fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner {
        let def = AttrDefId::ModuleId(self.root_module().id);
        AttrsWithOwner::new(db, def)
    }
    fn attr_id(self) -> AttrDefId {
        AttrDefId::ModuleId(self.root_module().id)
    }
}

/// Resolves the item `link` points to in the scope of `def`.
pub fn resolve_doc_path_on(
    db: &dyn HirDatabase,
    def: impl HasAttrs + Copy,
    link: &str,
    ns: Option<Namespace>,
    is_inner_doc: bool,
) -> Option<DocLinkDef> {
    resolve_doc_path_on_(db, link, def.attr_id(), ns, is_inner_doc)
}

fn resolve_doc_path_on_(
    db: &dyn HirDatabase,
    link: &str,
    attr_id: AttrDefId,
    ns: Option<Namespace>,
    is_inner_doc: bool,
) -> Option<DocLinkDef> {
    let resolver = match attr_id {
        AttrDefId::ModuleId(it) => {
            if is_inner_doc {
                it.resolver(db)
            } else if let Some(parent) = Module::from(it).parent(db) {
                parent.id.resolver(db)
            } else {
                it.resolver(db)
            }
        }
        AttrDefId::FieldId(it) => it.parent.resolver(db),
        AttrDefId::AdtId(it) => it.resolver(db),
        AttrDefId::FunctionId(it) => it.resolver(db),
        AttrDefId::EnumVariantId(it) => it.resolver(db),
        AttrDefId::StaticId(it) => it.resolver(db),
        AttrDefId::ConstId(it) => it.resolver(db),
        AttrDefId::TraitId(it) => it.resolver(db),
        AttrDefId::TraitAliasId(it) => it.resolver(db),
        AttrDefId::TypeAliasId(it) => it.resolver(db),
        AttrDefId::ImplId(it) => it.resolver(db),
        AttrDefId::ExternBlockId(it) => it.resolver(db),
        AttrDefId::UseId(it) => it.resolver(db),
        AttrDefId::MacroId(it) => it.resolver(db),
        AttrDefId::ExternCrateId(it) => it.resolver(db),
        AttrDefId::GenericParamId(_) => return None,
    };

    let mut modpath = doc_modpath_from_str(link)?;

    let resolved = resolver.resolve_module_path_in_items(db, &modpath);
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
    resolver: Resolver<'_>,
    path: ModPath,
    name: Name,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    let path = Path::from_known_path_with_no_generic(path);
    // FIXME: This does not handle `Self` on trait definitions, which we should resolve to the
    // trait itself.
    let base_def = resolver.resolve_path_in_type_ns_fully(db, &path)?;

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
            return id.trait_items(db).items.iter().find(|it| it.0 == name).map(|(_, assoc_id)| {
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
        TypeNs::ModuleId(_) => {
            return None;
        }
    };

    // Resolve inherent items first, then trait items, then fields.
    if let Some(assoc_item_def) = resolve_assoc_item(db, &ty, &name, ns) {
        return Some(assoc_item_def);
    }

    if let Some(impl_trait_item_def) = resolve_impl_trait_item(db, resolver, &ty, &name, ns) {
        return Some(impl_trait_item_def);
    }

    let variant_def = match ty.as_adt()? {
        Adt::Struct(it) => it.into(),
        Adt::Union(it) => it.into(),
        Adt::Enum(_) => return None,
    };
    resolve_field(db, variant_def, name, ns)
}

fn resolve_assoc_item<'db>(
    db: &'db dyn HirDatabase,
    ty: &Type<'db>,
    name: &Name,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    ty.iterate_assoc_items(db, ty.krate(db), move |assoc_item| {
        if assoc_item.name(db)? != *name {
            return None;
        }
        as_module_def_if_namespace_matches(assoc_item, ns)
    })
}

fn resolve_impl_trait_item<'db>(
    db: &'db dyn HirDatabase,
    resolver: Resolver<'_>,
    ty: &Type<'db>,
    name: &Name,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    let canonical = ty.canonical();
    let krate = ty.krate(db);
    let environment = resolver
        .generic_def()
        .map_or_else(|| crate::TraitEnvironment::empty(krate.id), |d| db.trait_environment(d));
    let traits_in_scope = resolver.traits_in_scope(db);

    let mut result = None;

    // `ty.iterate_path_candidates()` require a scope, which is not available when resolving
    // attributes here. Use path resolution directly instead.
    //
    // FIXME: resolve type aliases (which are not yielded by iterate_path_candidates)
    _ = method_resolution::iterate_path_candidates(
        &canonical,
        db,
        environment,
        &traits_in_scope,
        method_resolution::VisibleFromModule::None,
        Some(name),
        &mut |_, assoc_item_id: AssocItemId, _| {
            // If two traits in scope define the same item, Rustdoc links to no specific trait (for
            // instance, given two methods `a`, Rustdoc simply links to `method.a` with no
            // disambiguation) so we just pick the first one we find as well.
            result = as_module_def_if_namespace_matches(assoc_item_id.into(), ns);

            if result.is_some() { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
        },
    );

    result
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

fn as_module_def_if_namespace_matches(
    assoc_item: AssocItem,
    ns: Option<Namespace>,
) -> Option<DocLinkDef> {
    let (def, expected_ns) = match assoc_item {
        AssocItem::Function(it) => (ModuleDef::Function(it), Namespace::Values),
        AssocItem::Const(it) => (ModuleDef::Const(it), Namespace::Values),
        AssocItem::TypeAlias(it) => (ModuleDef::TypeAlias(it), Namespace::Types),
    };

    (ns.unwrap_or(expected_ns) == expected_ns).then_some(DocLinkDef::ModuleDef(def))
}

fn doc_modpath_from_str(link: &str) -> Option<ModPath> {
    // FIXME: this is not how we should get a mod path here.
    let try_get_modpath = |link: &str| {
        let mut parts = link.split("::");
        let mut first_segment = None;
        let kind = match parts.next()? {
            "" => PathKind::Abs,
            "crate" => PathKind::Crate,
            "self" => PathKind::SELF,
            "super" => {
                let mut deg = 1;
                for segment in parts.by_ref() {
                    if segment == "super" {
                        deg += 1;
                    } else {
                        first_segment = Some(segment);
                        break;
                    }
                }
                PathKind::Super(deg)
            }
            segment => {
                first_segment = Some(segment);
                PathKind::Plain
            }
        };
        let parts = first_segment.into_iter().chain(parts).map(|segment| match segment.parse() {
            Ok(idx) => Name::new_tuple_field(idx),
            Err(_) => Name::new_root(segment.split_once('<').map_or(segment, |it| it.0)),
        });
        Some(ModPath::from_segments(kind, parts))
    };
    try_get_modpath(link)
}
