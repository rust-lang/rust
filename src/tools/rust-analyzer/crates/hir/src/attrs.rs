//! Attributes & documentation for hir types.

use cfg::CfgExpr;
use either::Either;
use hir_def::{
    AssocItemId, AttrDefId, FieldId, LifetimeParamId, ModuleDefId, TypeOrConstParamId,
    attrs::{AttrFlags, Docs, IsInnerDoc},
    expr_store::path::Path,
    item_scope::ItemInNs,
    per_ns::Namespace,
    resolver::{HasResolver, Resolver, TypeNs},
};
use hir_expand::{
    mod_path::{ModPath, PathKind},
    name::Name,
};
use hir_ty::{
    db::HirDatabase,
    method_resolution::{
        self, CandidateId, MethodError, MethodResolutionContext, MethodResolutionUnstableFeatures,
    },
    next_solver::{DbInterner, TypingMode, infer::DbInternerInferExt},
};
use intern::Symbol;

use crate::{
    Adt, AsAssocItem, AssocItem, BuiltinType, Const, ConstParam, DocLinkDef, Enum, ExternCrateDecl,
    Field, Function, GenericParam, HasCrate, Impl, LangItem, LifetimeParam, Macro, Module,
    ModuleDef, Static, Struct, Trait, Type, TypeAlias, TypeParam, Union, Variant, VariantDef,
};

#[derive(Debug, Clone, Copy)]
pub enum AttrsOwner {
    AttrDef(AttrDefId),
    Field(FieldId),
    LifetimeParam(LifetimeParamId),
    TypeOrConstParam(TypeOrConstParamId),
}

impl AttrsOwner {
    #[inline]
    fn attr_def(&self) -> Option<AttrDefId> {
        match self {
            AttrsOwner::AttrDef(it) => Some(*it),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttrsWithOwner {
    pub(crate) attrs: AttrFlags,
    owner: AttrsOwner,
}

impl AttrsWithOwner {
    fn new(db: &dyn HirDatabase, owner: AttrDefId) -> Self {
        Self { attrs: AttrFlags::query(db, owner), owner: AttrsOwner::AttrDef(owner) }
    }

    fn new_field(db: &dyn HirDatabase, owner: FieldId) -> Self {
        Self { attrs: AttrFlags::query_field(db, owner), owner: AttrsOwner::Field(owner) }
    }

    fn new_lifetime_param(db: &dyn HirDatabase, owner: LifetimeParamId) -> Self {
        Self {
            attrs: AttrFlags::query_lifetime_param(db, owner),
            owner: AttrsOwner::LifetimeParam(owner),
        }
    }
    fn new_type_or_const_param(db: &dyn HirDatabase, owner: TypeOrConstParamId) -> Self {
        Self {
            attrs: AttrFlags::query_type_or_const_param(db, owner),
            owner: AttrsOwner::TypeOrConstParam(owner),
        }
    }

    #[inline]
    pub fn is_unstable(&self) -> bool {
        self.attrs.contains(AttrFlags::IS_UNSTABLE)
    }

    #[inline]
    pub fn is_macro_export(&self) -> bool {
        self.attrs.contains(AttrFlags::IS_MACRO_EXPORT)
    }

    #[inline]
    pub fn is_doc_notable_trait(&self) -> bool {
        self.attrs.contains(AttrFlags::IS_DOC_NOTABLE_TRAIT)
    }

    #[inline]
    pub fn is_doc_hidden(&self) -> bool {
        self.attrs.contains(AttrFlags::IS_DOC_HIDDEN)
    }

    #[inline]
    pub fn is_deprecated(&self) -> bool {
        self.attrs.contains(AttrFlags::IS_DEPRECATED)
    }

    #[inline]
    pub fn is_non_exhaustive(&self) -> bool {
        self.attrs.contains(AttrFlags::NON_EXHAUSTIVE)
    }

    #[inline]
    pub fn is_test(&self) -> bool {
        self.attrs.contains(AttrFlags::IS_TEST)
    }

    #[inline]
    pub fn lang(&self, db: &dyn HirDatabase) -> Option<LangItem> {
        self.owner
            .attr_def()
            .and_then(|owner| self.attrs.lang_item_with_attrs(db, owner))
            .and_then(|lang| LangItem::from_symbol(&lang))
    }

    #[inline]
    pub fn doc_aliases<'db>(&self, db: &'db dyn HirDatabase) -> &'db [Symbol] {
        let owner = match self.owner {
            AttrsOwner::AttrDef(it) => Either::Left(it),
            AttrsOwner::Field(it) => Either::Right(it),
            AttrsOwner::LifetimeParam(_) | AttrsOwner::TypeOrConstParam(_) => return &[],
        };
        self.attrs.doc_aliases(db, owner)
    }

    #[inline]
    pub fn cfgs<'db>(&self, db: &'db dyn HirDatabase) -> Option<&'db CfgExpr> {
        let owner = match self.owner {
            AttrsOwner::AttrDef(it) => Either::Left(it),
            AttrsOwner::Field(it) => Either::Right(it),
            AttrsOwner::LifetimeParam(_) | AttrsOwner::TypeOrConstParam(_) => return None,
        };
        self.attrs.cfgs(db, owner)
    }

    #[inline]
    pub fn hir_docs<'db>(&self, db: &'db dyn HirDatabase) -> Option<&'db Docs> {
        match self.owner {
            AttrsOwner::AttrDef(it) => AttrFlags::docs(db, it).as_deref(),
            AttrsOwner::Field(it) => AttrFlags::field_docs(db, it),
            AttrsOwner::LifetimeParam(_) | AttrsOwner::TypeOrConstParam(_) => None,
        }
    }
}

pub trait HasAttrs: Sized {
    #[inline]
    fn attrs(self, db: &dyn HirDatabase) -> AttrsWithOwner {
        match self.attr_id(db) {
            AttrsOwner::AttrDef(it) => AttrsWithOwner::new(db, it),
            AttrsOwner::Field(it) => AttrsWithOwner::new_field(db, it),
            AttrsOwner::LifetimeParam(it) => AttrsWithOwner::new_lifetime_param(db, it),
            AttrsOwner::TypeOrConstParam(it) => AttrsWithOwner::new_type_or_const_param(db, it),
        }
    }

    #[doc(hidden)]
    fn attr_id(self, db: &dyn HirDatabase) -> AttrsOwner;

    #[inline]
    fn hir_docs(self, db: &dyn HirDatabase) -> Option<&Docs> {
        match self.attr_id(db) {
            AttrsOwner::AttrDef(it) => AttrFlags::docs(db, it).as_deref(),
            AttrsOwner::Field(it) => AttrFlags::field_docs(db, it),
            AttrsOwner::LifetimeParam(_) | AttrsOwner::TypeOrConstParam(_) => None,
        }
    }
}

macro_rules! impl_has_attrs {
    ($(($def:ident, $def_id:ident),)*) => {$(
        impl HasAttrs for $def {
            #[inline]
            fn attr_id(self, _db: &dyn HirDatabase) -> AttrsOwner {
                AttrsOwner::AttrDef(AttrDefId::$def_id(self.into()))
            }
        }
    )*};
}

impl_has_attrs![
    (Variant, EnumVariantId),
    (Static, StaticId),
    (Const, ConstId),
    (Trait, TraitId),
    (TypeAlias, TypeAliasId),
    (Macro, MacroId),
    (Function, FunctionId),
    (Adt, AdtId),
    (Impl, ImplId),
    (ExternCrateDecl, ExternCrateId),
];

macro_rules! impl_has_attrs_enum {
    ($($variant:ident),* for $enum:ident) => {$(
        impl HasAttrs for $variant {
            #[inline]
            fn attr_id(self, db: &dyn HirDatabase) -> AttrsOwner {
                $enum::$variant(self).attr_id(db)
            }
        }
    )*};
}

impl_has_attrs_enum![Struct, Union, Enum for Adt];
impl_has_attrs_enum![TypeParam, ConstParam, LifetimeParam for GenericParam];

impl HasAttrs for Module {
    #[inline]
    fn attr_id(self, _: &dyn HirDatabase) -> AttrsOwner {
        AttrsOwner::AttrDef(AttrDefId::ModuleId(self.id))
    }
}

impl HasAttrs for GenericParam {
    #[inline]
    fn attr_id(self, _db: &dyn HirDatabase) -> AttrsOwner {
        match self {
            GenericParam::TypeParam(it) => AttrsOwner::TypeOrConstParam(it.merge().into()),
            GenericParam::ConstParam(it) => AttrsOwner::TypeOrConstParam(it.merge().into()),
            GenericParam::LifetimeParam(it) => AttrsOwner::LifetimeParam(it.into()),
        }
    }
}

impl HasAttrs for AssocItem {
    #[inline]
    fn attr_id(self, db: &dyn HirDatabase) -> AttrsOwner {
        match self {
            AssocItem::Function(it) => it.attr_id(db),
            AssocItem::Const(it) => it.attr_id(db),
            AssocItem::TypeAlias(it) => it.attr_id(db),
        }
    }
}

impl HasAttrs for crate::Crate {
    #[inline]
    fn attr_id(self, db: &dyn HirDatabase) -> AttrsOwner {
        self.root_module(db).attr_id(db)
    }
}

impl HasAttrs for Field {
    #[inline]
    fn attr_id(self, _db: &dyn HirDatabase) -> AttrsOwner {
        AttrsOwner::Field(self.into())
    }
}

/// Resolves the item `link` points to in the scope of `def`.
pub fn resolve_doc_path_on(
    db: &dyn HirDatabase,
    def: impl HasAttrs + Copy,
    link: &str,
    ns: Option<Namespace>,
    is_inner_doc: IsInnerDoc,
) -> Option<DocLinkDef> {
    resolve_doc_path_on_(db, link, def.attr_id(db), ns, is_inner_doc)
}

fn resolve_doc_path_on_(
    db: &dyn HirDatabase,
    link: &str,
    attr_id: AttrsOwner,
    ns: Option<Namespace>,
    is_inner_doc: IsInnerDoc,
) -> Option<DocLinkDef> {
    let resolver = match attr_id {
        AttrsOwner::AttrDef(AttrDefId::ModuleId(it)) => {
            if is_inner_doc.yes() {
                it.resolver(db)
            } else if let Some(parent) = Module::from(it).parent(db) {
                parent.id.resolver(db)
            } else {
                it.resolver(db)
            }
        }
        AttrsOwner::AttrDef(AttrDefId::AdtId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::FunctionId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::EnumVariantId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::StaticId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::ConstId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::TraitId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::TypeAliasId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::ImplId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::ExternBlockId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::UseId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::MacroId(it)) => it.resolver(db),
        AttrsOwner::AttrDef(AttrDefId::ExternCrateId(it)) => it.resolver(db),
        AttrsOwner::Field(it) => it.parent.resolver(db),
        AttrsOwner::LifetimeParam(_) | AttrsOwner::TypeOrConstParam(_) => return None,
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
    ty.iterate_assoc_items(db, move |assoc_item| {
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
    let krate = ty.krate(db);
    let environment = crate::param_env_from_resolver(db, &resolver);
    let traits_in_scope = resolver.traits_in_scope(db);

    // `ty.iterate_path_candidates()` require a scope, which is not available when resolving
    // attributes here. Use path resolution directly instead.
    //
    // FIXME: resolve type aliases (which are not yielded by iterate_path_candidates)
    let interner = DbInterner::new_with(db, environment.krate);
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let unstable_features =
        MethodResolutionUnstableFeatures::from_def_map(resolver.top_level_def_map());
    let ctx = MethodResolutionContext {
        infcx: &infcx,
        resolver: &resolver,
        param_env: environment.param_env,
        traits_in_scope: &traits_in_scope,
        edition: krate.edition(db),
        unstable_features: &unstable_features,
    };
    let resolution = ctx.probe_for_name(method_resolution::Mode::Path, name.clone(), ty.ty);
    let resolution = match resolution {
        Ok(resolution) => resolution.item,
        Err(MethodError::PrivateMatch(resolution)) => resolution.item,
        _ => return None,
    };
    let resolution = match resolution {
        CandidateId::FunctionId(id) => AssocItem::Function(id.into()),
        CandidateId::ConstId(id) => AssocItem::Const(id.into()),
    };
    as_module_def_if_namespace_matches(resolution, ns)
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
