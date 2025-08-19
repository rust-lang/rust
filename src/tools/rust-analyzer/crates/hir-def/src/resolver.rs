//! Name resolution façade.
use std::{fmt, mem};

use base_db::Crate;
use hir_expand::{
    MacroDefId,
    mod_path::{ModPath, PathKind},
    name::{AsName, Name},
};
use intern::{Symbol, sym};
use itertools::Itertools as _;
use rustc_hash::FxHashSet;
use smallvec::{SmallVec, smallvec};
use span::SyntaxContext;
use syntax::ast::HasName;
use triomphe::Arc;

use crate::{
    AdtId, AstIdLoc, ConstId, ConstParamId, CrateRootModuleId, DefWithBodyId, EnumId,
    EnumVariantId, ExternBlockId, ExternCrateId, FunctionId, FxIndexMap, GenericDefId,
    GenericParamId, HasModule, ImplId, ItemContainerId, LifetimeParamId, LocalModuleId, Lookup,
    Macro2Id, MacroId, MacroRulesId, ModuleDefId, ModuleId, ProcMacroId, StaticId, StructId,
    TraitAliasId, TraitId, TypeAliasId, TypeOrConstParamId, TypeParamId, UseId, VariantId,
    builtin_type::BuiltinType,
    db::DefDatabase,
    expr_store::{
        HygieneId,
        path::Path,
        scope::{ExprScopes, ScopeId},
    },
    hir::{
        BindingId, ExprId, LabelId,
        generics::{GenericParams, TypeOrConstParamData},
    },
    item_scope::{BUILTIN_SCOPE, BuiltinShadowMode, ImportOrExternCrate, ImportOrGlob, ItemScope},
    lang_item::LangItemTarget,
    nameres::{DefMap, LocalDefMap, MacroSubNs, ResolvePathResultPrefixInfo, block_def_map},
    per_ns::PerNs,
    src::HasSource,
    type_ref::LifetimeRef,
    visibility::{RawVisibility, Visibility},
};

#[derive(Debug, Clone)]
pub struct Resolver<'db> {
    /// The stack of scopes, where the inner-most scope is the last item.
    ///
    /// When using, you generally want to process the scopes in reverse order,
    /// there's `scopes` *method* for that.
    scopes: Vec<Scope<'db>>,
    module_scope: ModuleItemMap<'db>,
}

#[derive(Clone)]
struct ModuleItemMap<'db> {
    def_map: &'db DefMap,
    local_def_map: &'db LocalDefMap,
    module_id: LocalModuleId,
}

impl fmt::Debug for ModuleItemMap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleItemMap").field("module_id", &self.module_id).finish()
    }
}

#[derive(Clone)]
struct ExprScope {
    owner: DefWithBodyId,
    expr_scopes: Arc<ExprScopes>,
    scope_id: ScopeId,
}

impl fmt::Debug for ExprScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExprScope")
            .field("owner", &self.owner)
            .field("scope_id", &self.scope_id)
            .finish()
    }
}

#[derive(Debug, Clone)]
enum Scope<'db> {
    /// All the items and imported names of a module
    BlockScope(ModuleItemMap<'db>),
    /// Brings the generic parameters of an item into scope as well as the `Self` type alias /
    /// generic for ADTs and impls.
    GenericParams { def: GenericDefId, params: Arc<GenericParams> },
    /// Local bindings
    ExprScope(ExprScope),
    /// Macro definition inside bodies that affects all paths after it in the same block.
    MacroDefScope(MacroDefId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeNs {
    SelfType(ImplId),
    GenericParam(TypeParamId),
    AdtId(AdtId),
    AdtSelfType(AdtId),
    // Yup, enum variants are added to the types ns, but any usage of variant as
    // type is an error.
    EnumVariantId(EnumVariantId),
    TypeAliasId(TypeAliasId),
    BuiltinType(BuiltinType),
    TraitId(TraitId),
    TraitAliasId(TraitAliasId),

    ModuleId(ModuleId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolveValueResult {
    ValueNs(ValueNs, Option<ImportOrGlob>),
    Partial(TypeNs, usize, Option<ImportOrExternCrate>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ValueNs {
    ImplSelf(ImplId),
    LocalBinding(BindingId),
    FunctionId(FunctionId),
    ConstId(ConstId),
    StaticId(StaticId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
    GenericParam(ConstParamId),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum LifetimeNs {
    Static,
    LifetimeParam(LifetimeParamId),
}

impl<'db> Resolver<'db> {
    /// Resolve known trait from std, like `std::futures::Future`
    pub fn resolve_known_trait(&self, db: &dyn DefDatabase, path: &ModPath) -> Option<TraitId> {
        let res = self.resolve_module_path(db, path, BuiltinShadowMode::Other).take_types()?;
        match res {
            ModuleDefId::TraitId(it) => Some(it),
            _ => None,
        }
    }

    /// Resolve known struct from std, like `std::boxed::Box`
    pub fn resolve_known_struct(&self, db: &dyn DefDatabase, path: &ModPath) -> Option<StructId> {
        let res = self.resolve_module_path(db, path, BuiltinShadowMode::Other).take_types()?;
        match res {
            ModuleDefId::AdtId(AdtId::StructId(it)) => Some(it),
            _ => None,
        }
    }

    /// Resolve known enum from std, like `std::result::Result`
    pub fn resolve_known_enum(&self, db: &dyn DefDatabase, path: &ModPath) -> Option<EnumId> {
        let res = self.resolve_module_path(db, path, BuiltinShadowMode::Other).take_types()?;
        match res {
            ModuleDefId::AdtId(AdtId::EnumId(it)) => Some(it),
            _ => None,
        }
    }

    pub fn resolve_module_path_in_items(&self, db: &dyn DefDatabase, path: &ModPath) -> PerNs {
        self.resolve_module_path(db, path, BuiltinShadowMode::Module)
    }

    pub fn resolve_path_in_type_ns(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
    ) -> Option<(TypeNs, Option<usize>, Option<ImportOrExternCrate>)> {
        self.resolve_path_in_type_ns_with_prefix_info(db, path).map(
            |(resolution, remaining_segments, import, _)| (resolution, remaining_segments, import),
        )
    }

    pub fn resolve_path_in_type_ns_with_prefix_info(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
    ) -> Option<(TypeNs, Option<usize>, Option<ImportOrExternCrate>, ResolvePathResultPrefixInfo)>
    {
        let path = match path {
            Path::BarePath(mod_path) => mod_path,
            Path::Normal(it) => &it.mod_path,
            Path::LangItem(l, seg) => {
                let type_ns = match *l {
                    LangItemTarget::Union(it) => TypeNs::AdtId(it.into()),
                    LangItemTarget::TypeAlias(it) => TypeNs::TypeAliasId(it),
                    LangItemTarget::Struct(it) => TypeNs::AdtId(it.into()),
                    LangItemTarget::EnumVariant(it) => TypeNs::EnumVariantId(it),
                    LangItemTarget::EnumId(it) => TypeNs::AdtId(it.into()),
                    LangItemTarget::Trait(it) => TypeNs::TraitId(it),
                    LangItemTarget::Function(_)
                    | LangItemTarget::ImplDef(_)
                    | LangItemTarget::Static(_) => return None,
                };
                return Some((
                    type_ns,
                    seg.as_ref().map(|_| 1),
                    None,
                    ResolvePathResultPrefixInfo::default(),
                ));
            }
        };
        let first_name = path.segments().first()?;
        let skip_to_mod = path.kind != PathKind::Plain;
        if skip_to_mod {
            return self.module_scope.resolve_path_in_type_ns(db, path);
        }

        let remaining_idx = || {
            if path.segments().len() == 1 { None } else { Some(1) }
        };

        for scope in self.scopes() {
            match scope {
                Scope::ExprScope(_) | Scope::MacroDefScope(_) => continue,
                Scope::GenericParams { params, def } => {
                    if let &GenericDefId::ImplId(impl_) = def {
                        if *first_name == sym::Self_ {
                            return Some((
                                TypeNs::SelfType(impl_),
                                remaining_idx(),
                                None,
                                ResolvePathResultPrefixInfo::default(),
                            ));
                        }
                    } else if let &GenericDefId::AdtId(adt) = def
                        && *first_name == sym::Self_
                    {
                        return Some((
                            TypeNs::AdtSelfType(adt),
                            remaining_idx(),
                            None,
                            ResolvePathResultPrefixInfo::default(),
                        ));
                    }
                    if let Some(id) = params.find_type_by_name(first_name, *def) {
                        return Some((
                            TypeNs::GenericParam(id),
                            remaining_idx(),
                            None,
                            ResolvePathResultPrefixInfo::default(),
                        ));
                    }
                }
                Scope::BlockScope(m) => {
                    if let Some(res) = m.resolve_path_in_type_ns(db, path) {
                        let res = match res.0 {
                            TypeNs::ModuleId(_) if res.1.is_none() => {
                                if let Some(ModuleDefId::BuiltinType(builtin)) = BUILTIN_SCOPE
                                    .get(first_name)
                                    .and_then(|builtin| builtin.take_types())
                                {
                                    (
                                        TypeNs::BuiltinType(builtin),
                                        remaining_idx(),
                                        None,
                                        ResolvePathResultPrefixInfo::default(),
                                    )
                                } else {
                                    res
                                }
                            }
                            _ => res,
                        };
                        return Some(res);
                    }
                }
            }
        }
        self.module_scope.resolve_path_in_type_ns(db, path)
    }

    pub fn resolve_path_in_type_ns_fully(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
    ) -> Option<TypeNs> {
        let (res, unresolved, _) = self.resolve_path_in_type_ns(db, path)?;
        if unresolved.is_some() {
            return None;
        }
        Some(res)
    }

    pub fn resolve_visibility(
        &self,
        db: &dyn DefDatabase,
        visibility: &RawVisibility,
    ) -> Option<Visibility> {
        match visibility {
            RawVisibility::Module(_, _) => {
                let (item_map, item_local_map, module) = self.item_scope_();
                item_map.resolve_visibility(
                    item_local_map,
                    db,
                    module,
                    visibility,
                    self.scopes().any(|scope| {
                        matches!(scope, Scope::GenericParams { def: GenericDefId::ImplId(_), .. })
                    }),
                )
            }
            RawVisibility::PubSelf(explicitness) => {
                Some(Visibility::Module(self.module(), *explicitness))
            }
            RawVisibility::PubCrate => Some(Visibility::PubCrate(self.krate())),
            RawVisibility::Public => Some(Visibility::Public),
        }
    }

    pub fn resolve_path_in_value_ns(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
        hygiene_id: HygieneId,
    ) -> Option<ResolveValueResult> {
        self.resolve_path_in_value_ns_with_prefix_info(db, path, hygiene_id).map(|(it, _)| it)
    }

    pub fn resolve_path_in_value_ns_with_prefix_info(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
        mut hygiene_id: HygieneId,
    ) -> Option<(ResolveValueResult, ResolvePathResultPrefixInfo)> {
        let path = match path {
            Path::BarePath(mod_path) => mod_path,
            Path::Normal(it) => &it.mod_path,
            Path::LangItem(l, None) => {
                return Some((
                    ResolveValueResult::ValueNs(
                        match *l {
                            LangItemTarget::Function(it) => ValueNs::FunctionId(it),
                            LangItemTarget::Static(it) => ValueNs::StaticId(it),
                            LangItemTarget::Struct(it) => ValueNs::StructId(it),
                            LangItemTarget::EnumVariant(it) => ValueNs::EnumVariantId(it),
                            LangItemTarget::Union(_)
                            | LangItemTarget::ImplDef(_)
                            | LangItemTarget::TypeAlias(_)
                            | LangItemTarget::Trait(_)
                            | LangItemTarget::EnumId(_) => return None,
                        },
                        None,
                    ),
                    ResolvePathResultPrefixInfo::default(),
                ));
            }
            Path::LangItem(l, Some(_)) => {
                let type_ns = match *l {
                    LangItemTarget::Union(it) => TypeNs::AdtId(it.into()),
                    LangItemTarget::TypeAlias(it) => TypeNs::TypeAliasId(it),
                    LangItemTarget::Struct(it) => TypeNs::AdtId(it.into()),
                    LangItemTarget::EnumVariant(it) => TypeNs::EnumVariantId(it),
                    LangItemTarget::EnumId(it) => TypeNs::AdtId(it.into()),
                    LangItemTarget::Trait(it) => TypeNs::TraitId(it),
                    LangItemTarget::Function(_)
                    | LangItemTarget::ImplDef(_)
                    | LangItemTarget::Static(_) => return None,
                };
                // Remaining segments start from 0 because lang paths have no segments other than the remaining.
                return Some((
                    ResolveValueResult::Partial(type_ns, 0, None),
                    ResolvePathResultPrefixInfo::default(),
                ));
            }
        };
        let n_segments = path.segments().len();
        let tmp = Name::new_symbol_root(sym::self_);
        let first_name = if path.is_self() { &tmp } else { path.segments().first()? };
        let skip_to_mod = path.kind != PathKind::Plain && !path.is_self();
        if skip_to_mod {
            return self.module_scope.resolve_path_in_value_ns(db, path);
        }

        if n_segments <= 1 {
            let mut hygiene_info = hygiene_info(db, hygiene_id);
            for scope in self.scopes() {
                match scope {
                    Scope::ExprScope(scope) => {
                        let entry =
                            scope.expr_scopes.entries(scope.scope_id).iter().find(|entry| {
                                entry.name() == first_name && entry.hygiene() == hygiene_id
                            });

                        if let Some(e) = entry {
                            return Some((
                                ResolveValueResult::ValueNs(
                                    ValueNs::LocalBinding(e.binding()),
                                    None,
                                ),
                                ResolvePathResultPrefixInfo::default(),
                            ));
                        }
                    }
                    Scope::MacroDefScope(macro_id) => {
                        handle_macro_def_scope(db, &mut hygiene_id, &mut hygiene_info, macro_id)
                    }
                    Scope::GenericParams { params, def } => {
                        if let &GenericDefId::ImplId(impl_) = def
                            && *first_name == sym::Self_
                        {
                            return Some((
                                ResolveValueResult::ValueNs(ValueNs::ImplSelf(impl_), None),
                                ResolvePathResultPrefixInfo::default(),
                            ));
                        }
                        if let Some(id) = params.find_const_by_name(first_name, *def) {
                            let val = ValueNs::GenericParam(id);
                            return Some((
                                ResolveValueResult::ValueNs(val, None),
                                ResolvePathResultPrefixInfo::default(),
                            ));
                        }
                    }
                    Scope::BlockScope(m) => {
                        if let Some(def) = m.resolve_path_in_value_ns(db, path) {
                            return Some(def);
                        }
                    }
                }
            }
        } else {
            for scope in self.scopes() {
                match scope {
                    Scope::ExprScope(_) | Scope::MacroDefScope(_) => continue,
                    Scope::GenericParams { params, def } => {
                        if let &GenericDefId::ImplId(impl_) = def {
                            if *first_name == sym::Self_ {
                                return Some((
                                    ResolveValueResult::Partial(TypeNs::SelfType(impl_), 1, None),
                                    ResolvePathResultPrefixInfo::default(),
                                ));
                            }
                        } else if let &GenericDefId::AdtId(adt) = def
                            && *first_name == sym::Self_
                        {
                            let ty = TypeNs::AdtSelfType(adt);
                            return Some((
                                ResolveValueResult::Partial(ty, 1, None),
                                ResolvePathResultPrefixInfo::default(),
                            ));
                        }
                        if let Some(id) = params.find_type_by_name(first_name, *def) {
                            let ty = TypeNs::GenericParam(id);
                            return Some((
                                ResolveValueResult::Partial(ty, 1, None),
                                ResolvePathResultPrefixInfo::default(),
                            ));
                        }
                    }
                    Scope::BlockScope(m) => {
                        if let Some(def) = m.resolve_path_in_value_ns(db, path) {
                            return Some(def);
                        }
                    }
                }
            }
        }

        if let Some(res) = self.module_scope.resolve_path_in_value_ns(db, path) {
            return Some(res);
        }

        // If a path of the shape `u16::from_le_bytes` failed to resolve at all, then we fall back
        // to resolving to the primitive type, to allow this to still work in the presence of
        // `use core::u16;`.
        if path.kind == PathKind::Plain
            && n_segments > 1
            && let Some(builtin) = BuiltinType::by_name(first_name)
        {
            return Some((
                ResolveValueResult::Partial(TypeNs::BuiltinType(builtin), 1, None),
                ResolvePathResultPrefixInfo::default(),
            ));
        }

        None
    }

    pub fn resolve_path_in_value_ns_fully(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
        hygiene: HygieneId,
    ) -> Option<ValueNs> {
        match self.resolve_path_in_value_ns(db, path, hygiene)? {
            ResolveValueResult::ValueNs(it, _) => Some(it),
            ResolveValueResult::Partial(..) => None,
        }
    }

    pub fn resolve_path_as_macro(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
        expected_macro_kind: Option<MacroSubNs>,
    ) -> Option<(MacroId, Option<ImportOrExternCrate>)> {
        let (item_map, item_local_map, module) = self.item_scope_();
        item_map
            .resolve_path(
                item_local_map,
                db,
                module,
                path,
                BuiltinShadowMode::Other,
                expected_macro_kind,
            )
            .0
            .take_macros_import()
    }

    pub fn resolve_path_as_macro_def(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
        expected_macro_kind: Option<MacroSubNs>,
    ) -> Option<MacroDefId> {
        self.resolve_path_as_macro(db, path, expected_macro_kind).map(|(it, _)| db.macro_def(it))
    }

    pub fn resolve_lifetime(&self, lifetime: &LifetimeRef) -> Option<LifetimeNs> {
        match lifetime {
            LifetimeRef::Static => Some(LifetimeNs::Static),
            LifetimeRef::Named(name) => self.scopes().find_map(|scope| match scope {
                Scope::GenericParams { def, params } => {
                    params.find_lifetime_by_name(name, *def).map(LifetimeNs::LifetimeParam)
                }
                _ => None,
            }),
            LifetimeRef::Placeholder | LifetimeRef::Error => None,
            LifetimeRef::Param(lifetime_param_id) => {
                Some(LifetimeNs::LifetimeParam(*lifetime_param_id))
            }
        }
    }

    /// Returns a set of names available in the current scope.
    ///
    /// Note that this is a somewhat fuzzy concept -- internally, the compiler
    /// doesn't necessary follow a strict scoping discipline. Rather, it just
    /// tells for each ident what it resolves to.
    ///
    /// A good example is something like `str::from_utf8`. From scopes point of
    /// view, this code is erroneous -- both `str` module and `str` type occupy
    /// the same type namespace.
    ///
    /// We don't try to model that super-correctly -- this functionality is
    /// primarily exposed for completions.
    ///
    /// Note that in Rust one name can be bound to several items:
    ///
    /// ```
    /// # #![allow(non_camel_case_types)]
    /// macro_rules! t { () => (()) }
    /// type t = t!();
    /// const t: t = t!();
    /// ```
    ///
    /// That's why we return a multimap.
    ///
    /// The shadowing is accounted for: in
    ///
    /// ```ignore
    /// let it = 92;
    /// {
    ///     let it = 92;
    ///     $0
    /// }
    /// ```
    ///
    /// there will be only one entry for `it` in the result.
    ///
    /// The result is ordered *roughly* from the innermost scope to the
    /// outermost: when the name is introduced in two namespaces in two scopes,
    /// we use the position of the first scope.
    pub fn names_in_scope(
        &self,
        db: &dyn DefDatabase,
    ) -> FxIndexMap<Name, SmallVec<[ScopeDef; 1]>> {
        let mut res = ScopeNames::default();
        for scope in self.scopes() {
            scope.process_names(&mut res, db);
        }
        let ModuleItemMap { def_map, module_id, local_def_map } = self.module_scope;
        // FIXME: should we provide `self` here?
        // f(
        //     Name::self_param(),
        //     PerNs::types(Resolution::Def {
        //         def: m.module.into(),
        //     }),
        // );
        def_map[module_id].scope.entries().for_each(|(name, def)| {
            res.add_per_ns(name, def);
        });

        def_map[module_id].scope.legacy_macros().for_each(|(name, macs)| {
            macs.iter().for_each(|&mac| {
                res.add(name, ScopeDef::ModuleDef(ModuleDefId::MacroId(mac)));
            })
        });
        def_map.macro_use_prelude().iter().sorted_by_key(|&(k, _)| k.clone()).for_each(
            |(name, &(def, _extern_crate))| {
                res.add(name, ScopeDef::ModuleDef(def.into()));
            },
        );
        local_def_map.extern_prelude().for_each(|(name, (def, _extern_crate))| {
            res.add(name, ScopeDef::ModuleDef(ModuleDefId::ModuleId(def.into())));
        });
        BUILTIN_SCOPE.iter().for_each(|(name, &def)| {
            res.add_per_ns(name, def);
        });
        if let Some((prelude, _use)) = def_map.prelude() {
            let prelude_def_map = prelude.def_map(db);
            for (name, def) in prelude_def_map[prelude.local_id].scope.entries() {
                res.add_per_ns(name, def)
            }
        }
        res.map
    }

    /// Note: Not to be used directly within hir-def/hir-ty
    pub fn extern_crate_decls_in_scope<'a>(
        &'a self,
        db: &'a dyn DefDatabase,
    ) -> impl Iterator<Item = Name> + 'a {
        self.module_scope.def_map[self.module_scope.module_id]
            .scope
            .extern_crate_decls()
            .filter_map(|id| {
                let loc = id.lookup(db);
                let extern_crate = loc.source(db);
                // If there is a rename (`as x`), extract the renamed name, or remove the `extern crate`
                // if it is an underscore.
                extern_crate
                    .value
                    .rename()
                    .map(|a| a.name().map(|it| it.as_name()))
                    .unwrap_or_else(|| extern_crate.value.name_ref().map(|it| it.as_name()))
            })
    }

    pub fn extern_crates_in_scope(&self) -> impl Iterator<Item = (Name, ModuleId)> + '_ {
        self.module_scope
            .local_def_map
            .extern_prelude()
            .map(|(name, module_id)| (name.clone(), module_id.0.into()))
    }

    pub fn traits_in_scope(&self, db: &dyn DefDatabase) -> FxHashSet<TraitId> {
        // FIXME(trait_alias): Trait alias brings aliased traits in scope! Note that supertraits of
        // aliased traits are NOT brought in scope (unless also aliased).
        let mut traits = FxHashSet::default();

        for scope in self.scopes() {
            match scope {
                Scope::BlockScope(m) => traits.extend(m.def_map[m.module_id].scope.traits()),
                &Scope::GenericParams { def: GenericDefId::ImplId(impl_), .. } => {
                    let impl_data = db.impl_signature(impl_);
                    if let Some(target_trait) = impl_data.target_trait
                        && let Some(TypeNs::TraitId(trait_)) = self
                            .resolve_path_in_type_ns_fully(db, &impl_data.store[target_trait.path])
                    {
                        traits.insert(trait_);
                    }
                }
                _ => (),
            }
        }

        // Fill in the prelude traits
        if let Some((prelude, _use)) = self.module_scope.def_map.prelude() {
            let prelude_def_map = prelude.def_map(db);
            traits.extend(prelude_def_map[prelude.local_id].scope.traits());
        }
        // Fill in module visible traits
        traits.extend(self.module_scope.def_map[self.module_scope.module_id].scope.traits());
        traits
    }

    pub fn traits_in_scope_from_block_scopes(&self) -> impl Iterator<Item = TraitId> + '_ {
        self.scopes()
            .filter_map(|scope| match scope {
                Scope::BlockScope(m) => Some(m.def_map[m.module_id].scope.traits()),
                _ => None,
            })
            .flatten()
    }

    pub fn module(&self) -> ModuleId {
        let (def_map, _, local_id) = self.item_scope_();
        def_map.module_id(local_id)
    }

    pub fn item_scope(&self) -> &ItemScope {
        let (def_map, _, local_id) = self.item_scope_();
        &def_map[local_id].scope
    }

    pub fn krate(&self) -> Crate {
        self.module_scope.def_map.krate()
    }

    pub fn def_map(&self) -> &DefMap {
        self.item_scope_().0
    }

    pub fn generic_def(&self) -> Option<GenericDefId> {
        self.scopes().find_map(|scope| match scope {
            Scope::GenericParams { def, .. } => Some(*def),
            _ => None,
        })
    }

    pub fn generic_params(&self) -> Option<&GenericParams> {
        self.scopes().find_map(|scope| match scope {
            Scope::GenericParams { params, .. } => Some(&**params),
            _ => None,
        })
    }

    pub fn all_generic_params(&self) -> impl Iterator<Item = (&GenericParams, &GenericDefId)> {
        self.scopes().filter_map(|scope| match scope {
            Scope::GenericParams { params, def } => Some((&**params, def)),
            _ => None,
        })
    }

    pub fn body_owner(&self) -> Option<DefWithBodyId> {
        self.scopes().find_map(|scope| match scope {
            Scope::ExprScope(it) => Some(it.owner),
            _ => None,
        })
    }

    pub fn impl_def(&self) -> Option<ImplId> {
        self.scopes().find_map(|scope| match scope {
            &Scope::GenericParams { def: GenericDefId::ImplId(def), .. } => Some(def),
            _ => None,
        })
    }

    /// Checks if we rename `renamed` (currently named `current_name`) to `new_name`, will the meaning of this reference
    /// (that contains `current_name` path) change from `renamed` to some another variable (returned as `Some`).
    pub fn rename_will_conflict_with_another_variable(
        &self,
        db: &dyn DefDatabase,
        current_name: &Name,
        current_name_as_path: &ModPath,
        mut hygiene_id: HygieneId,
        new_name: &Symbol,
        to_be_renamed: BindingId,
    ) -> Option<BindingId> {
        let mut hygiene_info = hygiene_info(db, hygiene_id);
        let mut will_be_resolved_to = None;
        for scope in self.scopes() {
            match scope {
                Scope::ExprScope(scope) => {
                    for entry in scope.expr_scopes.entries(scope.scope_id) {
                        if entry.hygiene() == hygiene_id {
                            if entry.binding() == to_be_renamed {
                                // This currently resolves to our renamed variable, now `will_be_resolved_to`
                                // contains `Some` if the meaning will change or `None` if not.
                                return will_be_resolved_to;
                            } else if entry.name().symbol() == new_name {
                                will_be_resolved_to = Some(entry.binding());
                            }
                        }
                    }
                }
                Scope::MacroDefScope(macro_id) => {
                    handle_macro_def_scope(db, &mut hygiene_id, &mut hygiene_info, macro_id)
                }
                Scope::GenericParams { params, def } => {
                    if params.find_const_by_name(current_name, *def).is_some() {
                        // It does not resolve to our renamed variable.
                        return None;
                    }
                }
                Scope::BlockScope(m) => {
                    if m.resolve_path_in_value_ns(db, current_name_as_path).is_some() {
                        // It does not resolve to our renamed variable.
                        return None;
                    }
                }
            }
        }
        // It does not resolve to our renamed variable.
        None
    }

    /// Checks if we rename `renamed` to `name`, will the meaning of this reference (that contains `name` path) change
    /// from some other variable (returned as `Some`) to `renamed`.
    pub fn rename_will_conflict_with_renamed(
        &self,
        db: &dyn DefDatabase,
        name: &Name,
        name_as_path: &ModPath,
        mut hygiene_id: HygieneId,
        to_be_renamed: BindingId,
    ) -> Option<BindingId> {
        let mut hygiene_info = hygiene_info(db, hygiene_id);
        let mut will_resolve_to_renamed = false;
        for scope in self.scopes() {
            match scope {
                Scope::ExprScope(scope) => {
                    for entry in scope.expr_scopes.entries(scope.scope_id) {
                        if entry.binding() == to_be_renamed {
                            will_resolve_to_renamed = true;
                        } else if entry.hygiene() == hygiene_id && entry.name() == name {
                            if will_resolve_to_renamed {
                                // This will resolve to the renamed variable before it resolves to the original variable.
                                return Some(entry.binding());
                            } else {
                                // This will resolve to the original variable.
                                return None;
                            }
                        }
                    }
                }
                Scope::MacroDefScope(macro_id) => {
                    handle_macro_def_scope(db, &mut hygiene_id, &mut hygiene_info, macro_id)
                }
                Scope::GenericParams { params, def } => {
                    if params.find_const_by_name(name, *def).is_some() {
                        // Here and below, it might actually resolve to our renamed variable - in which case it'll
                        // hide the generic parameter or some other thing (not a variable). We don't check for that
                        // because due to naming conventions, it is rare that variable will shadow a non-variable.
                        return None;
                    }
                }
                Scope::BlockScope(m) => {
                    if m.resolve_path_in_value_ns(db, name_as_path).is_some() {
                        return None;
                    }
                }
            }
        }
        None
    }

    /// `expr_id` is required to be an expression id that comes after the top level expression scope in the given resolver
    #[must_use]
    pub fn update_to_inner_scope(
        &mut self,
        db: &'db dyn DefDatabase,
        owner: DefWithBodyId,
        expr_id: ExprId,
    ) -> UpdateGuard {
        #[inline(always)]
        fn append_expr_scope<'db>(
            db: &'db dyn DefDatabase,
            resolver: &mut Resolver<'db>,
            owner: DefWithBodyId,
            expr_scopes: &Arc<ExprScopes>,
            scope_id: ScopeId,
        ) {
            if let Some(macro_id) = expr_scopes.macro_def(scope_id) {
                resolver.scopes.push(Scope::MacroDefScope(**macro_id));
            }
            resolver.scopes.push(Scope::ExprScope(ExprScope {
                owner,
                expr_scopes: expr_scopes.clone(),
                scope_id,
            }));
            if let Some(block) = expr_scopes.block(scope_id) {
                let def_map = block_def_map(db, block);
                let local_def_map = block.lookup(db).module.only_local_def_map(db);
                resolver.scopes.push(Scope::BlockScope(ModuleItemMap {
                    def_map,
                    local_def_map,
                    module_id: DefMap::ROOT,
                }));
                // FIXME: This adds as many module scopes as there are blocks, but resolving in each
                // already traverses all parents, so this is O(n²). I think we could only store the
                // innermost module scope instead?
            }
        }

        let start = self.scopes.len();
        let innermost_scope = self.scopes().find(|scope| !matches!(scope, Scope::MacroDefScope(_)));
        match innermost_scope {
            Some(&Scope::ExprScope(ExprScope { scope_id, ref expr_scopes, owner })) => {
                let expr_scopes = expr_scopes.clone();
                let scope_chain = expr_scopes
                    .scope_chain(expr_scopes.scope_for(expr_id))
                    .take_while(|&it| it != scope_id);
                for scope_id in scope_chain {
                    append_expr_scope(db, self, owner, &expr_scopes, scope_id);
                }
            }
            _ => {
                let expr_scopes = db.expr_scopes(owner);
                let scope_chain = expr_scopes.scope_chain(expr_scopes.scope_for(expr_id));

                for scope_id in scope_chain {
                    append_expr_scope(db, self, owner, &expr_scopes, scope_id);
                }
            }
        }
        self.scopes[start..].reverse();
        UpdateGuard(start)
    }

    pub fn reset_to_guard(&mut self, UpdateGuard(start): UpdateGuard) {
        self.scopes.truncate(start);
    }
}

#[inline]
fn handle_macro_def_scope(
    db: &dyn DefDatabase,
    hygiene_id: &mut HygieneId,
    hygiene_info: &mut Option<(SyntaxContext, MacroDefId)>,
    macro_id: &MacroDefId,
) {
    if let Some((parent_ctx, label_macro_id)) = hygiene_info
        && label_macro_id == macro_id
    {
        // A macro is allowed to refer to variables from before its declaration.
        // Therefore, if we got to the rib of its declaration, give up its hygiene
        // and use its parent expansion.
        *hygiene_id = HygieneId::new(parent_ctx.opaque_and_semitransparent(db));
        *hygiene_info = parent_ctx.outer_expn(db).map(|expansion| {
            let expansion = db.lookup_intern_macro_call(expansion.into());
            (parent_ctx.parent(db), expansion.def)
        });
    }
}

#[inline]
fn hygiene_info(
    db: &dyn DefDatabase,
    hygiene_id: HygieneId,
) -> Option<(SyntaxContext, MacroDefId)> {
    if !hygiene_id.is_root() {
        let ctx = hygiene_id.lookup();
        ctx.outer_expn(db).map(|expansion| {
            let expansion = db.lookup_intern_macro_call(expansion.into());
            (ctx.parent(db), expansion.def)
        })
    } else {
        None
    }
}

pub struct UpdateGuard(usize);

impl<'db> Resolver<'db> {
    fn scopes(&self) -> impl Iterator<Item = &Scope<'db>> {
        self.scopes.iter().rev()
    }

    fn resolve_module_path(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> PerNs {
        let (item_map, item_local_map, module) = self.item_scope_();
        // This method resolves `path` just like import paths, so no expected macro subns is given.
        let (module_res, segment_index) =
            item_map.resolve_path(item_local_map, db, module, path, shadow, None);
        if segment_index.is_some() {
            return PerNs::none();
        }
        module_res
    }

    /// The innermost block scope that contains items or the module scope that contains this resolver.
    fn item_scope_(&self) -> (&DefMap, &LocalDefMap, LocalModuleId) {
        self.scopes()
            .find_map(|scope| match scope {
                Scope::BlockScope(m) => Some((m.def_map, m.local_def_map, m.module_id)),
                _ => None,
            })
            .unwrap_or((
                self.module_scope.def_map,
                self.module_scope.local_def_map,
                self.module_scope.module_id,
            ))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScopeDef {
    ModuleDef(ModuleDefId),
    Unknown,
    ImplSelfType(ImplId),
    AdtSelfType(AdtId),
    GenericParam(GenericParamId),
    Local(BindingId),
    Label(LabelId),
}

impl<'db> Scope<'db> {
    fn process_names(&self, acc: &mut ScopeNames, db: &'db dyn DefDatabase) {
        match self {
            Scope::BlockScope(m) => {
                m.def_map[m.module_id].scope.entries().for_each(|(name, def)| {
                    acc.add_per_ns(name, def);
                });
                m.def_map[m.module_id].scope.legacy_macros().for_each(|(name, macs)| {
                    macs.iter().for_each(|&mac| {
                        acc.add(name, ScopeDef::ModuleDef(ModuleDefId::MacroId(mac)));
                    })
                });
            }
            &Scope::GenericParams { ref params, def: parent } => {
                if let GenericDefId::ImplId(impl_) = parent {
                    acc.add(&Name::new_symbol_root(sym::Self_), ScopeDef::ImplSelfType(impl_));
                } else if let GenericDefId::AdtId(adt) = parent {
                    acc.add(&Name::new_symbol_root(sym::Self_), ScopeDef::AdtSelfType(adt));
                }

                for (local_id, param) in params.iter_type_or_consts() {
                    if let Some(name) = &param.name() {
                        let id = TypeOrConstParamId { parent, local_id };
                        let data = &db.generic_params(parent)[local_id];
                        acc.add(
                            name,
                            ScopeDef::GenericParam(match data {
                                TypeOrConstParamData::TypeParamData(_) => {
                                    GenericParamId::TypeParamId(TypeParamId::from_unchecked(id))
                                }
                                TypeOrConstParamData::ConstParamData(_) => {
                                    GenericParamId::ConstParamId(ConstParamId::from_unchecked(id))
                                }
                            }),
                        );
                    }
                }
                for (local_id, param) in params.iter_lt() {
                    let id = LifetimeParamId { parent, local_id };
                    acc.add(&param.name, ScopeDef::GenericParam(id.into()))
                }
            }
            Scope::ExprScope(scope) => {
                if let Some((label, name)) = scope.expr_scopes.label(scope.scope_id) {
                    acc.add(&name, ScopeDef::Label(label))
                }
                scope.expr_scopes.entries(scope.scope_id).iter().for_each(|e| {
                    acc.add_local(e.name(), e.binding());
                });
            }
            Scope::MacroDefScope(_) => {}
        }
    }
}

pub fn resolver_for_scope(
    db: &dyn DefDatabase,
    owner: DefWithBodyId,
    scope_id: Option<ScopeId>,
) -> Resolver<'_> {
    let r = owner.resolver(db);
    let scopes = db.expr_scopes(owner);
    resolver_for_scope_(db, scopes, scope_id, r, owner)
}

fn resolver_for_scope_<'db>(
    db: &'db dyn DefDatabase,
    scopes: Arc<ExprScopes>,
    scope_id: Option<ScopeId>,
    mut r: Resolver<'db>,
    owner: DefWithBodyId,
) -> Resolver<'db> {
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    r.scopes.reserve(scope_chain.len());

    for scope in scope_chain.into_iter().rev() {
        if let Some(block) = scopes.block(scope) {
            let def_map = block_def_map(db, block);
            let local_def_map = block.lookup(db).module.only_local_def_map(db);
            r = r.push_block_scope(def_map, local_def_map);
            // FIXME: This adds as many module scopes as there are blocks, but resolving in each
            // already traverses all parents, so this is O(n²). I think we could only store the
            // innermost module scope instead?
        }
        if let Some(macro_id) = scopes.macro_def(scope) {
            r = r.push_scope(Scope::MacroDefScope(**macro_id));
        }

        r = r.push_expr_scope(owner, Arc::clone(&scopes), scope);
    }
    r
}

impl<'db> Resolver<'db> {
    fn push_scope(mut self, scope: Scope<'db>) -> Resolver<'db> {
        self.scopes.push(scope);
        self
    }

    fn push_generic_params_scope(
        self,
        db: &'db dyn DefDatabase,
        def: GenericDefId,
    ) -> Resolver<'db> {
        let params = db.generic_params(def);
        self.push_scope(Scope::GenericParams { def, params })
    }

    fn push_block_scope(
        self,
        def_map: &'db DefMap,
        local_def_map: &'db LocalDefMap,
    ) -> Resolver<'db> {
        self.push_scope(Scope::BlockScope(ModuleItemMap {
            def_map,
            local_def_map,
            module_id: DefMap::ROOT,
        }))
    }

    fn push_expr_scope(
        self,
        owner: DefWithBodyId,
        expr_scopes: Arc<ExprScopes>,
        scope_id: ScopeId,
    ) -> Resolver<'db> {
        self.push_scope(Scope::ExprScope(ExprScope { owner, expr_scopes, scope_id }))
    }
}

impl<'db> ModuleItemMap<'db> {
    fn resolve_path_in_value_ns(
        &self,
        db: &'db dyn DefDatabase,
        path: &ModPath,
    ) -> Option<(ResolveValueResult, ResolvePathResultPrefixInfo)> {
        let (module_def, unresolved_idx, prefix_info) = self.def_map.resolve_path_locally(
            self.local_def_map,
            db,
            self.module_id,
            path,
            BuiltinShadowMode::Other,
        );
        match unresolved_idx {
            None => {
                let (value, import) = to_value_ns(module_def)?;
                Some((ResolveValueResult::ValueNs(value, import), prefix_info))
            }
            Some(unresolved_idx) => {
                let def = module_def.take_types_full()?;
                let ty = match def.def {
                    ModuleDefId::AdtId(it) => TypeNs::AdtId(it),
                    ModuleDefId::TraitId(it) => TypeNs::TraitId(it),
                    ModuleDefId::TraitAliasId(it) => TypeNs::TraitAliasId(it),
                    ModuleDefId::TypeAliasId(it) => TypeNs::TypeAliasId(it),
                    ModuleDefId::BuiltinType(it) => TypeNs::BuiltinType(it),

                    ModuleDefId::ModuleId(_)
                    | ModuleDefId::FunctionId(_)
                    | ModuleDefId::EnumVariantId(_)
                    | ModuleDefId::ConstId(_)
                    | ModuleDefId::MacroId(_)
                    | ModuleDefId::StaticId(_) => return None,
                };
                Some((ResolveValueResult::Partial(ty, unresolved_idx, def.import), prefix_info))
            }
        }
    }

    fn resolve_path_in_type_ns(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
    ) -> Option<(TypeNs, Option<usize>, Option<ImportOrExternCrate>, ResolvePathResultPrefixInfo)>
    {
        let (module_def, idx, prefix_info) = self.def_map.resolve_path_locally(
            self.local_def_map,
            db,
            self.module_id,
            path,
            BuiltinShadowMode::Other,
        );
        let (res, import) = to_type_ns(module_def)?;
        Some((res, idx, import, prefix_info))
    }
}

fn to_value_ns(per_ns: PerNs) -> Option<(ValueNs, Option<ImportOrGlob>)> {
    let (def, import) = per_ns.take_values_import()?;
    let res = match def {
        ModuleDefId::FunctionId(it) => ValueNs::FunctionId(it),
        ModuleDefId::AdtId(AdtId::StructId(it)) => ValueNs::StructId(it),
        ModuleDefId::EnumVariantId(it) => ValueNs::EnumVariantId(it),
        ModuleDefId::ConstId(it) => ValueNs::ConstId(it),
        ModuleDefId::StaticId(it) => ValueNs::StaticId(it),

        ModuleDefId::AdtId(AdtId::EnumId(_) | AdtId::UnionId(_))
        | ModuleDefId::TraitId(_)
        | ModuleDefId::TraitAliasId(_)
        | ModuleDefId::TypeAliasId(_)
        | ModuleDefId::BuiltinType(_)
        | ModuleDefId::MacroId(_)
        | ModuleDefId::ModuleId(_) => return None,
    };
    Some((res, import))
}

fn to_type_ns(per_ns: PerNs) -> Option<(TypeNs, Option<ImportOrExternCrate>)> {
    let def = per_ns.take_types_full()?;
    let res = match def.def {
        ModuleDefId::AdtId(it) => TypeNs::AdtId(it),
        ModuleDefId::EnumVariantId(it) => TypeNs::EnumVariantId(it),

        ModuleDefId::TypeAliasId(it) => TypeNs::TypeAliasId(it),
        ModuleDefId::BuiltinType(it) => TypeNs::BuiltinType(it),

        ModuleDefId::TraitId(it) => TypeNs::TraitId(it),
        ModuleDefId::TraitAliasId(it) => TypeNs::TraitAliasId(it),

        ModuleDefId::ModuleId(it) => TypeNs::ModuleId(it),

        ModuleDefId::FunctionId(_)
        | ModuleDefId::ConstId(_)
        | ModuleDefId::MacroId(_)
        | ModuleDefId::StaticId(_) => return None,
    };
    Some((res, def.import))
}

#[derive(Default)]
struct ScopeNames {
    map: FxIndexMap<Name, SmallVec<[ScopeDef; 1]>>,
}

impl ScopeNames {
    fn add(&mut self, name: &Name, def: ScopeDef) {
        let set = self.map.entry(name.clone()).or_default();
        if !set.contains(&def) {
            set.push(def)
        }
    }
    fn add_per_ns(&mut self, name: &Name, def: PerNs) {
        if let Some(ty) = &def.types {
            self.add(name, ScopeDef::ModuleDef(ty.def))
        }
        if let Some(def) = &def.values {
            self.add(name, ScopeDef::ModuleDef(def.def))
        }
        if let Some(mac) = &def.macros {
            self.add(name, ScopeDef::ModuleDef(ModuleDefId::MacroId(mac.def)))
        }
        if def.is_none() {
            self.add(name, ScopeDef::Unknown)
        }
    }
    fn add_local(&mut self, name: &Name, binding: BindingId) {
        let set = self.map.entry(name.clone()).or_default();
        // XXX: hack, account for local (and only local) shadowing.
        //
        // This should be somewhat more principled and take namespaces into
        // accounts, but, alas, scoping rules are a hoax. `str` type and `str`
        // module can be both available in the same scope.
        if set.iter().any(|it| matches!(it, &ScopeDef::Local(_))) {
            cov_mark::hit!(shadowing_shows_single_completion);
            return;
        }
        set.push(ScopeDef::Local(binding))
    }
}

pub trait HasResolver: Copy {
    /// Builds a resolver for type references inside this def.
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_>;
}

impl HasResolver for ModuleId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        let (mut def_map, local_def_map) = self.local_def_map(db);
        let mut module_id = self.local_id;

        if !self.is_block_module() {
            return Resolver {
                scopes: vec![],
                module_scope: ModuleItemMap { def_map, local_def_map, module_id },
            };
        }

        let mut modules: SmallVec<[_; 1]> = smallvec![];
        while let Some(parent) = def_map.parent() {
            let block_def_map = mem::replace(&mut def_map, parent.def_map(db));
            modules.push(block_def_map);
            if !parent.is_block_module() {
                module_id = parent.local_id;
                break;
            }
        }
        let mut resolver = Resolver {
            scopes: Vec::with_capacity(modules.len()),
            module_scope: ModuleItemMap { def_map, local_def_map, module_id },
        };
        for def_map in modules.into_iter().rev() {
            resolver = resolver.push_block_scope(def_map, local_def_map);
        }
        resolver
    }
}

impl HasResolver for CrateRootModuleId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        let (def_map, local_def_map) = self.local_def_map(db);
        Resolver {
            scopes: vec![],
            module_scope: ModuleItemMap { def_map, local_def_map, module_id: DefMap::ROOT },
        }
    }
}

impl HasResolver for TraitId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for TraitAliasId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self).push_generic_params_scope(db, self.into())
    }
}

impl<T: Into<AdtId> + Copy> HasResolver for T {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        let def = self.into();
        def.module(db).resolver(db).push_generic_params_scope(db, def.into())
    }
}

impl HasResolver for FunctionId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for ConstId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self)
    }
}

impl HasResolver for StaticId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self)
    }
}

impl HasResolver for TypeAliasId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for ImplId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        self.lookup(db).container.resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for ExternBlockId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        // Same as parent's
        lookup_resolver(db, self)
    }
}

impl HasResolver for ExternCrateId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self)
    }
}

impl HasResolver for UseId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self)
    }
}

impl HasResolver for DefWithBodyId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        match self {
            DefWithBodyId::ConstId(c) => c.resolver(db),
            DefWithBodyId::FunctionId(f) => f.resolver(db),
            DefWithBodyId::StaticId(s) => s.resolver(db),
            DefWithBodyId::VariantId(v) => v.resolver(db),
        }
    }
}

impl HasResolver for ItemContainerId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        match self {
            ItemContainerId::ModuleId(it) => it.resolver(db),
            ItemContainerId::TraitId(it) => it.resolver(db),
            ItemContainerId::ImplId(it) => it.resolver(db),
            ItemContainerId::ExternBlockId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for GenericDefId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        match self {
            GenericDefId::FunctionId(inner) => inner.resolver(db),
            GenericDefId::AdtId(adt) => adt.resolver(db),
            GenericDefId::TraitId(inner) => inner.resolver(db),
            GenericDefId::TraitAliasId(inner) => inner.resolver(db),
            GenericDefId::TypeAliasId(inner) => inner.resolver(db),
            GenericDefId::ImplId(inner) => inner.resolver(db),
            GenericDefId::ConstId(inner) => inner.resolver(db),
            GenericDefId::StaticId(inner) => inner.resolver(db),
        }
    }
}

impl HasResolver for EnumVariantId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        self.lookup(db).parent.resolver(db)
    }
}

impl HasResolver for VariantId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        match self {
            VariantId::EnumVariantId(it) => it.resolver(db),
            VariantId::StructId(it) => it.resolver(db),
            VariantId::UnionId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for MacroId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        match self {
            MacroId::Macro2Id(it) => it.resolver(db),
            MacroId::MacroRulesId(it) => it.resolver(db),
            MacroId::ProcMacroId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for Macro2Id {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self)
    }
}

impl HasResolver for ProcMacroId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self)
    }
}

impl HasResolver for MacroRulesId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver<'_> {
        lookup_resolver(db, self)
    }
}

fn lookup_resolver(
    db: &dyn DefDatabase,
    lookup: impl Lookup<Database = dyn DefDatabase, Data = impl AstIdLoc<Container = impl HasResolver>>,
) -> Resolver<'_> {
    lookup.lookup(db).container().resolver(db)
}
