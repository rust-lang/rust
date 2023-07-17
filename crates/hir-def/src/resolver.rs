//! Name resolution façade.
use std::{fmt, hash::BuildHasherDefault};

use base_db::CrateId;
use hir_expand::name::{name, Name};
use indexmap::IndexMap;
use intern::Interned;
use rustc_hash::FxHashSet;
use smallvec::{smallvec, SmallVec};
use triomphe::Arc;

use crate::{
    body::scope::{ExprScopes, ScopeId},
    builtin_type::BuiltinType,
    db::DefDatabase,
    generics::{GenericParams, TypeOrConstParamData},
    hir::{BindingId, ExprId, LabelId},
    item_scope::{BuiltinShadowMode, BUILTIN_SCOPE},
    lang_item::LangItemTarget,
    nameres::{DefMap, MacroSubNs},
    path::{ModPath, Path, PathKind},
    per_ns::PerNs,
    visibility::{RawVisibility, Visibility},
    AdtId, AssocItemId, ConstId, ConstParamId, CrateRootModuleId, DefWithBodyId, EnumId,
    EnumVariantId, ExternBlockId, ExternCrateId, FunctionId, GenericDefId, GenericParamId,
    HasModule, ImplId, ItemContainerId, LifetimeParamId, LocalModuleId, Lookup, Macro2Id, MacroId,
    MacroRulesId, ModuleDefId, ModuleId, ProcMacroId, StaticId, StructId, TraitAliasId, TraitId,
    TypeAliasId, TypeOrConstParamId, TypeOwnerId, TypeParamId, VariantId,
};

#[derive(Debug, Clone)]
pub struct Resolver {
    /// The stack of scopes, where the inner-most scope is the last item.
    ///
    /// When using, you generally want to process the scopes in reverse order,
    /// there's `scopes` *method* for that.
    scopes: Vec<Scope>,
    module_scope: ModuleItemMap,
}

#[derive(Clone)]
struct ModuleItemMap {
    def_map: Arc<DefMap>,
    module_id: LocalModuleId,
}

impl fmt::Debug for ModuleItemMap {
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
enum Scope {
    /// All the items and imported names of a module
    BlockScope(ModuleItemMap),
    /// Brings the generic parameters of an item into scope
    GenericParams { def: GenericDefId, params: Interned<GenericParams> },
    /// Brings `Self` in `impl` block into scope
    ImplDefScope(ImplId),
    /// Brings `Self` in enum, struct and union definitions into scope
    AdtScope(AdtId),
    /// Local bindings
    ExprScope(ExprScope),
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
    // Module belong to type ns, but the resolver is used when all module paths
    // are fully resolved.
    // ModuleId(ModuleId)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolveValueResult {
    ValueNs(ValueNs),
    Partial(TypeNs, usize),
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

impl Resolver {
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

    // FIXME: This shouldn't exist
    pub fn resolve_module_path_in_trait_assoc_items(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
    ) -> Option<PerNs> {
        let (item_map, module) = self.item_scope();
        let (module_res, idx) =
            item_map.resolve_path(db, module, path, BuiltinShadowMode::Module, None);
        match module_res.take_types()? {
            ModuleDefId::TraitId(it) => {
                let idx = idx?;
                let unresolved = &path.segments()[idx..];
                let assoc = match unresolved {
                    [it] => it,
                    _ => return None,
                };
                let &(_, assoc) = db.trait_data(it).items.iter().find(|(n, _)| n == assoc)?;
                Some(match assoc {
                    AssocItemId::FunctionId(it) => PerNs::values(it.into(), Visibility::Public),
                    AssocItemId::ConstId(it) => PerNs::values(it.into(), Visibility::Public),
                    AssocItemId::TypeAliasId(it) => PerNs::types(it.into(), Visibility::Public),
                })
            }
            _ => None,
        }
    }

    pub fn resolve_path_in_type_ns(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
    ) -> Option<(TypeNs, Option<usize>)> {
        let path = match path {
            Path::Normal { mod_path, .. } => mod_path,
            Path::LangItem(l) => {
                return Some((
                    match *l {
                        LangItemTarget::Union(it) => TypeNs::AdtId(it.into()),
                        LangItemTarget::TypeAlias(it) => TypeNs::TypeAliasId(it),
                        LangItemTarget::Struct(it) => TypeNs::AdtId(it.into()),
                        LangItemTarget::EnumVariant(it) => TypeNs::EnumVariantId(it),
                        LangItemTarget::EnumId(it) => TypeNs::AdtId(it.into()),
                        LangItemTarget::Trait(it) => TypeNs::TraitId(it),
                        LangItemTarget::Function(_)
                        | LangItemTarget::ImplDef(_)
                        | LangItemTarget::Static(_) => return None,
                    },
                    None,
                ))
            }
        };
        let first_name = path.segments().first()?;
        let skip_to_mod = path.kind != PathKind::Plain;
        if skip_to_mod {
            return self.module_scope.resolve_path_in_type_ns(db, path);
        }

        let remaining_idx = || if path.segments().len() == 1 { None } else { Some(1) };

        for scope in self.scopes() {
            match scope {
                Scope::ExprScope(_) => continue,
                Scope::GenericParams { params, def } => {
                    if let Some(id) = params.find_type_by_name(first_name, *def) {
                        return Some((TypeNs::GenericParam(id), remaining_idx()));
                    }
                }
                &Scope::ImplDefScope(impl_) => {
                    if first_name == &name![Self] {
                        return Some((TypeNs::SelfType(impl_), remaining_idx()));
                    }
                }
                &Scope::AdtScope(adt) => {
                    if first_name == &name![Self] {
                        return Some((TypeNs::AdtSelfType(adt), remaining_idx()));
                    }
                }
                Scope::BlockScope(m) => {
                    if let Some(res) = m.resolve_path_in_type_ns(db, path) {
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
        let (res, unresolved) = self.resolve_path_in_type_ns(db, path)?;
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
        let within_impl =
            self.scopes().find(|scope| matches!(scope, Scope::ImplDefScope(_))).is_some();
        match visibility {
            RawVisibility::Module(_) => {
                let (item_map, module) = self.item_scope();
                item_map.resolve_visibility(db, module, visibility, within_impl)
            }
            RawVisibility::Public => Some(Visibility::Public),
        }
    }

    pub fn resolve_path_in_value_ns(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
    ) -> Option<ResolveValueResult> {
        let path = match path {
            Path::Normal { mod_path, .. } => mod_path,
            Path::LangItem(l) => {
                return Some(ResolveValueResult::ValueNs(match *l {
                    LangItemTarget::Function(it) => ValueNs::FunctionId(it),
                    LangItemTarget::Static(it) => ValueNs::StaticId(it),
                    LangItemTarget::Struct(it) => ValueNs::StructId(it),
                    LangItemTarget::EnumVariant(it) => ValueNs::EnumVariantId(it),
                    LangItemTarget::Union(_)
                    | LangItemTarget::ImplDef(_)
                    | LangItemTarget::TypeAlias(_)
                    | LangItemTarget::Trait(_)
                    | LangItemTarget::EnumId(_) => return None,
                }))
            }
        };
        let n_segments = path.segments().len();
        let tmp = name![self];
        let first_name = if path.is_self() { &tmp } else { path.segments().first()? };
        let skip_to_mod = path.kind != PathKind::Plain && !path.is_self();
        if skip_to_mod {
            return self.module_scope.resolve_path_in_value_ns(db, path);
        }

        if n_segments <= 1 {
            for scope in self.scopes() {
                match scope {
                    Scope::ExprScope(scope) => {
                        let entry = scope
                            .expr_scopes
                            .entries(scope.scope_id)
                            .iter()
                            .find(|entry| entry.name() == first_name);

                        if let Some(e) = entry {
                            return Some(ResolveValueResult::ValueNs(ValueNs::LocalBinding(
                                e.binding(),
                            )));
                        }
                    }
                    Scope::GenericParams { params, def } => {
                        if let Some(id) = params.find_const_by_name(first_name, *def) {
                            let val = ValueNs::GenericParam(id);
                            return Some(ResolveValueResult::ValueNs(val));
                        }
                    }
                    &Scope::ImplDefScope(impl_) => {
                        if first_name == &name![Self] {
                            return Some(ResolveValueResult::ValueNs(ValueNs::ImplSelf(impl_)));
                        }
                    }
                    // bare `Self` doesn't work in the value namespace in a struct/enum definition
                    Scope::AdtScope(_) => continue,
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
                    Scope::ExprScope(_) => continue,
                    Scope::GenericParams { params, def } => {
                        if let Some(id) = params.find_type_by_name(first_name, *def) {
                            let ty = TypeNs::GenericParam(id);
                            return Some(ResolveValueResult::Partial(ty, 1));
                        }
                    }
                    &Scope::ImplDefScope(impl_) => {
                        if first_name == &name![Self] {
                            return Some(ResolveValueResult::Partial(TypeNs::SelfType(impl_), 1));
                        }
                    }
                    Scope::AdtScope(adt) => {
                        if first_name == &name![Self] {
                            let ty = TypeNs::AdtSelfType(*adt);
                            return Some(ResolveValueResult::Partial(ty, 1));
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
        if path.kind == PathKind::Plain && n_segments > 1 {
            if let Some(builtin) = BuiltinType::by_name(first_name) {
                return Some(ResolveValueResult::Partial(TypeNs::BuiltinType(builtin), 1));
            }
        }

        None
    }

    pub fn resolve_path_in_value_ns_fully(
        &self,
        db: &dyn DefDatabase,
        path: &Path,
    ) -> Option<ValueNs> {
        match self.resolve_path_in_value_ns(db, path)? {
            ResolveValueResult::ValueNs(it) => Some(it),
            ResolveValueResult::Partial(..) => None,
        }
    }

    pub fn resolve_path_as_macro(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
        expected_macro_kind: Option<MacroSubNs>,
    ) -> Option<MacroId> {
        let (item_map, module) = self.item_scope();
        item_map
            .resolve_path(db, module, path, BuiltinShadowMode::Other, expected_macro_kind)
            .0
            .take_macros()
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
    /// macro_rules! t { () => (()) }
    /// type t = t!();
    /// const t: t = t!()
    /// ```
    ///
    /// That's why we return a multimap.
    ///
    /// The shadowing is accounted for: in
    ///
    /// ```
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
        let ModuleItemMap { ref def_map, module_id } = self.module_scope;
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
        def_map.macro_use_prelude().for_each(|(name, def)| {
            res.add(name, ScopeDef::ModuleDef(def.into()));
        });
        def_map.extern_prelude().for_each(|(name, def)| {
            res.add(name, ScopeDef::ModuleDef(ModuleDefId::ModuleId(def)));
        });
        BUILTIN_SCOPE.iter().for_each(|(name, &def)| {
            res.add_per_ns(name, def);
        });
        if let Some(prelude) = def_map.prelude() {
            let prelude_def_map = prelude.def_map(db);
            for (name, def) in prelude_def_map[prelude.local_id].scope.entries() {
                res.add_per_ns(name, def)
            }
        }
        res.map
    }

    pub fn traits_in_scope(&self, db: &dyn DefDatabase) -> FxHashSet<TraitId> {
        // FIXME(trait_alias): Trait alias brings aliased traits in scope! Note that supertraits of
        // aliased traits are NOT brought in scope (unless also aliased).
        let mut traits = FxHashSet::default();

        for scope in self.scopes() {
            match scope {
                Scope::BlockScope(m) => traits.extend(m.def_map[m.module_id].scope.traits()),
                &Scope::ImplDefScope(impl_) => {
                    if let Some(target_trait) = &db.impl_data(impl_).target_trait {
                        if let Some(TypeNs::TraitId(trait_)) =
                            self.resolve_path_in_type_ns_fully(db, &target_trait.path)
                        {
                            traits.insert(trait_);
                        }
                    }
                }
                _ => (),
            }
        }

        // Fill in the prelude traits
        if let Some(prelude) = self.module_scope.def_map.prelude() {
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
        let (def_map, local_id) = self.item_scope();
        def_map.module_id(local_id)
    }

    pub fn krate(&self) -> CrateId {
        self.module_scope.def_map.krate()
    }

    pub fn def_map(&self) -> &DefMap {
        self.item_scope().0
    }

    pub fn where_predicates_in_scope(
        &self,
    ) -> impl Iterator<Item = &crate::generics::WherePredicate> {
        self.scopes()
            .filter_map(|scope| match scope {
                Scope::GenericParams { params, .. } => Some(params),
                _ => None,
            })
            .flat_map(|params| params.where_predicates.iter())
    }

    pub fn generic_def(&self) -> Option<GenericDefId> {
        self.scopes().find_map(|scope| match scope {
            Scope::GenericParams { def, .. } => Some(*def),
            _ => None,
        })
    }

    pub fn generic_params(&self) -> Option<&Interned<GenericParams>> {
        self.scopes().find_map(|scope| match scope {
            Scope::GenericParams { params, .. } => Some(params),
            _ => None,
        })
    }

    pub fn body_owner(&self) -> Option<DefWithBodyId> {
        self.scopes().find_map(|scope| match scope {
            Scope::ExprScope(it) => Some(it.owner),
            _ => None,
        })
    }
    /// `expr_id` is required to be an expression id that comes after the top level expression scope in the given resolver
    #[must_use]
    pub fn update_to_inner_scope(
        &mut self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        expr_id: ExprId,
    ) -> UpdateGuard {
        #[inline(always)]
        fn append_expr_scope(
            db: &dyn DefDatabase,
            resolver: &mut Resolver,
            owner: DefWithBodyId,
            expr_scopes: &Arc<ExprScopes>,
            scope_id: ScopeId,
        ) {
            resolver.scopes.push(Scope::ExprScope(ExprScope {
                owner,
                expr_scopes: expr_scopes.clone(),
                scope_id,
            }));
            if let Some(block) = expr_scopes.block(scope_id) {
                let def_map = db.block_def_map(block);
                resolver
                    .scopes
                    .push(Scope::BlockScope(ModuleItemMap { def_map, module_id: DefMap::ROOT }));
                // FIXME: This adds as many module scopes as there are blocks, but resolving in each
                // already traverses all parents, so this is O(n²). I think we could only store the
                // innermost module scope instead?
            }
        }

        let start = self.scopes.len();
        let innermost_scope = self.scopes().next();
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

pub struct UpdateGuard(usize);

impl Resolver {
    fn scopes(&self) -> impl Iterator<Item = &Scope> {
        self.scopes.iter().rev()
    }

    fn resolve_module_path(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
        shadow: BuiltinShadowMode,
    ) -> PerNs {
        let (item_map, module) = self.item_scope();
        // This method resolves `path` just like import paths, so no expected macro subns is given.
        let (module_res, segment_index) = item_map.resolve_path(db, module, path, shadow, None);
        if segment_index.is_some() {
            return PerNs::none();
        }
        module_res
    }

    /// The innermost block scope that contains items or the module scope that contains this resolver.
    fn item_scope(&self) -> (&DefMap, LocalModuleId) {
        self.scopes()
            .find_map(|scope| match scope {
                Scope::BlockScope(m) => Some((&*m.def_map, m.module_id)),
                _ => None,
            })
            .unwrap_or((&self.module_scope.def_map, self.module_scope.module_id))
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

impl Scope {
    fn process_names(&self, acc: &mut ScopeNames, db: &dyn DefDatabase) {
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
            Scope::GenericParams { params, def: parent } => {
                let parent = *parent;
                for (local_id, param) in params.type_or_consts.iter() {
                    if let Some(name) = &param.name() {
                        let id = TypeOrConstParamId { parent, local_id };
                        let data = &db.generic_params(parent).type_or_consts[local_id];
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
                for (local_id, param) in params.lifetimes.iter() {
                    let id = LifetimeParamId { parent, local_id };
                    acc.add(&param.name, ScopeDef::GenericParam(id.into()))
                }
            }
            Scope::ImplDefScope(i) => {
                acc.add(&name![Self], ScopeDef::ImplSelfType(*i));
            }
            Scope::AdtScope(i) => {
                acc.add(&name![Self], ScopeDef::AdtSelfType(*i));
            }
            Scope::ExprScope(scope) => {
                if let Some((label, name)) = scope.expr_scopes.label(scope.scope_id) {
                    acc.add(&name, ScopeDef::Label(label))
                }
                scope.expr_scopes.entries(scope.scope_id).iter().for_each(|e| {
                    acc.add_local(e.name(), e.binding());
                });
            }
        }
    }
}

pub fn resolver_for_expr(db: &dyn DefDatabase, owner: DefWithBodyId, expr_id: ExprId) -> Resolver {
    let r = owner.resolver(db);
    let scopes = db.expr_scopes(owner);
    let scope_id = scopes.scope_for(expr_id);
    resolver_for_scope_(db, scopes, scope_id, r, owner)
}

pub fn resolver_for_scope(
    db: &dyn DefDatabase,
    owner: DefWithBodyId,
    scope_id: Option<ScopeId>,
) -> Resolver {
    let r = owner.resolver(db);
    let scopes = db.expr_scopes(owner);
    resolver_for_scope_(db, scopes, scope_id, r, owner)
}

fn resolver_for_scope_(
    db: &dyn DefDatabase,
    scopes: Arc<ExprScopes>,
    scope_id: Option<ScopeId>,
    mut r: Resolver,
    owner: DefWithBodyId,
) -> Resolver {
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    r.scopes.reserve(scope_chain.len());

    for scope in scope_chain.into_iter().rev() {
        if let Some(block) = scopes.block(scope) {
            let def_map = db.block_def_map(block);
            r = r.push_block_scope(def_map, DefMap::ROOT);
            // FIXME: This adds as many module scopes as there are blocks, but resolving in each
            // already traverses all parents, so this is O(n²). I think we could only store the
            // innermost module scope instead?
        }

        r = r.push_expr_scope(owner, Arc::clone(&scopes), scope);
    }
    r
}

impl Resolver {
    fn push_scope(mut self, scope: Scope) -> Resolver {
        self.scopes.push(scope);
        self
    }

    fn push_generic_params_scope(self, db: &dyn DefDatabase, def: GenericDefId) -> Resolver {
        let params = db.generic_params(def);
        self.push_scope(Scope::GenericParams { def, params })
    }

    fn push_impl_def_scope(self, impl_def: ImplId) -> Resolver {
        self.push_scope(Scope::ImplDefScope(impl_def))
    }

    fn push_block_scope(self, def_map: Arc<DefMap>, module_id: LocalModuleId) -> Resolver {
        self.push_scope(Scope::BlockScope(ModuleItemMap { def_map, module_id }))
    }

    fn push_expr_scope(
        self,
        owner: DefWithBodyId,
        expr_scopes: Arc<ExprScopes>,
        scope_id: ScopeId,
    ) -> Resolver {
        self.push_scope(Scope::ExprScope(ExprScope { owner, expr_scopes, scope_id }))
    }
}

impl ModuleItemMap {
    fn resolve_path_in_value_ns(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
    ) -> Option<ResolveValueResult> {
        let (module_def, idx) =
            self.def_map.resolve_path_locally(db, self.module_id, path, BuiltinShadowMode::Other);
        match idx {
            None => {
                let value = to_value_ns(module_def)?;
                Some(ResolveValueResult::ValueNs(value))
            }
            Some(idx) => {
                let ty = match module_def.take_types()? {
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
                Some(ResolveValueResult::Partial(ty, idx))
            }
        }
    }

    fn resolve_path_in_type_ns(
        &self,
        db: &dyn DefDatabase,
        path: &ModPath,
    ) -> Option<(TypeNs, Option<usize>)> {
        let (module_def, idx) =
            self.def_map.resolve_path_locally(db, self.module_id, path, BuiltinShadowMode::Other);
        let res = to_type_ns(module_def)?;
        Some((res, idx))
    }
}

fn to_value_ns(per_ns: PerNs) -> Option<ValueNs> {
    let res = match per_ns.take_values()? {
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
    Some(res)
}

fn to_type_ns(per_ns: PerNs) -> Option<TypeNs> {
    let res = match per_ns.take_types()? {
        ModuleDefId::AdtId(it) => TypeNs::AdtId(it),
        ModuleDefId::EnumVariantId(it) => TypeNs::EnumVariantId(it),

        ModuleDefId::TypeAliasId(it) => TypeNs::TypeAliasId(it),
        ModuleDefId::BuiltinType(it) => TypeNs::BuiltinType(it),

        ModuleDefId::TraitId(it) => TypeNs::TraitId(it),
        ModuleDefId::TraitAliasId(it) => TypeNs::TraitAliasId(it),

        ModuleDefId::FunctionId(_)
        | ModuleDefId::ConstId(_)
        | ModuleDefId::MacroId(_)
        | ModuleDefId::StaticId(_)
        | ModuleDefId::ModuleId(_) => return None,
    };
    Some(res)
}

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<rustc_hash::FxHasher>>;
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
        if let &Some((ty, _)) = &def.types {
            self.add(name, ScopeDef::ModuleDef(ty))
        }
        if let &Some((def, _)) = &def.values {
            self.add(name, ScopeDef::ModuleDef(def))
        }
        if let &Some((mac, _)) = &def.macros {
            self.add(name, ScopeDef::ModuleDef(ModuleDefId::MacroId(mac)))
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
    fn resolver(self, db: &dyn DefDatabase) -> Resolver;
}

impl HasResolver for ModuleId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        let mut def_map = self.def_map(db);
        let mut modules: SmallVec<[_; 1]> = smallvec![];
        let mut module_id = self.local_id;
        while let Some(parent) = def_map.parent() {
            modules.push((def_map, module_id));
            def_map = parent.def_map(db);
            module_id = parent.local_id;
        }
        let mut resolver = Resolver {
            scopes: Vec::with_capacity(modules.len()),
            module_scope: ModuleItemMap { def_map, module_id },
        };
        for (def_map, module) in modules.into_iter().rev() {
            resolver = resolver.push_block_scope(def_map, module);
        }
        resolver
    }
}

impl HasResolver for CrateRootModuleId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        Resolver {
            scopes: vec![],
            module_scope: ModuleItemMap { def_map: self.def_map(db), module_id: DefMap::ROOT },
        }
    }
}

impl HasResolver for TraitId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for TraitAliasId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl<T: Into<AdtId> + Copy> HasResolver for T {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        let def = self.into();
        def.module(db)
            .resolver(db)
            .push_generic_params_scope(db, def.into())
            .push_scope(Scope::AdtScope(def))
    }
}

impl HasResolver for FunctionId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for ConstId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db)
    }
}

impl HasResolver for StaticId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db)
    }
}

impl HasResolver for TypeAliasId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for ImplId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db)
            .container
            .resolver(db)
            .push_generic_params_scope(db, self.into())
            .push_impl_def_scope(self)
    }
}

impl HasResolver for ExternBlockId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        // Same as parent's
        self.lookup(db).container.resolver(db)
    }
}

impl HasResolver for ExternCrateId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db)
    }
}

impl HasResolver for TypeOwnerId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        match self {
            TypeOwnerId::FunctionId(it) => it.resolver(db),
            TypeOwnerId::StaticId(it) => it.resolver(db),
            TypeOwnerId::ConstId(it) => it.resolver(db),
            TypeOwnerId::InTypeConstId(it) => it.lookup(db).owner.resolver(db),
            TypeOwnerId::AdtId(it) => it.resolver(db),
            TypeOwnerId::TraitId(it) => it.resolver(db),
            TypeOwnerId::TraitAliasId(it) => it.resolver(db),
            TypeOwnerId::TypeAliasId(it) => it.resolver(db),
            TypeOwnerId::ImplId(it) => it.resolver(db),
            TypeOwnerId::EnumVariantId(it) => it.resolver(db),
            TypeOwnerId::ModuleId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for DefWithBodyId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        match self {
            DefWithBodyId::ConstId(c) => c.resolver(db),
            DefWithBodyId::FunctionId(f) => f.resolver(db),
            DefWithBodyId::StaticId(s) => s.resolver(db),
            DefWithBodyId::VariantId(v) => v.parent.resolver(db),
            DefWithBodyId::InTypeConstId(c) => c.lookup(db).owner.resolver(db),
        }
    }
}

impl HasResolver for ItemContainerId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        match self {
            ItemContainerId::ModuleId(it) => it.resolver(db),
            ItemContainerId::TraitId(it) => it.resolver(db),
            ItemContainerId::ImplId(it) => it.resolver(db),
            ItemContainerId::ExternBlockId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for GenericDefId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        match self {
            GenericDefId::FunctionId(inner) => inner.resolver(db),
            GenericDefId::AdtId(adt) => adt.resolver(db),
            GenericDefId::TraitId(inner) => inner.resolver(db),
            GenericDefId::TraitAliasId(inner) => inner.resolver(db),
            GenericDefId::TypeAliasId(inner) => inner.resolver(db),
            GenericDefId::ImplId(inner) => inner.resolver(db),
            GenericDefId::EnumVariantId(inner) => inner.parent.resolver(db),
            GenericDefId::ConstId(inner) => inner.resolver(db),
        }
    }
}

impl HasResolver for EnumVariantId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.parent.resolver(db)
    }
}

impl HasResolver for VariantId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        match self {
            VariantId::EnumVariantId(it) => it.parent.resolver(db),
            VariantId::StructId(it) => it.resolver(db),
            VariantId::UnionId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for MacroId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        match self {
            MacroId::Macro2Id(it) => it.resolver(db),
            MacroId::MacroRulesId(it) => it.resolver(db),
            MacroId::ProcMacroId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for Macro2Id {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db)
    }
}

impl HasResolver for ProcMacroId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db)
    }
}

impl HasResolver for MacroRulesId {
    fn resolver(self, db: &dyn DefDatabase) -> Resolver {
        self.lookup(db).container.resolver(db)
    }
}
