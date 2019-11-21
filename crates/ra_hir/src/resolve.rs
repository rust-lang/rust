//! Name resolution.
use std::sync::Arc;

use hir_def::{
    body::scope::{ExprScopes, ScopeId},
    builtin_type::BuiltinType,
    db::DefDatabase2,
    expr::{ExprId, PatId},
    generics::GenericParams,
    nameres::{per_ns::PerNs, CrateDefMap},
    path::{Path, PathKind},
    AdtId, AstItemDef, ConstId, ContainerId, CrateModuleId, DefWithBodyId, EnumId, EnumVariantId,
    FunctionId, GenericDefId, ImplId, Lookup, ModuleDefId, ModuleId, StaticId, StructId, TraitId,
    TypeAliasId, UnionId,
};
use hir_expand::{
    name::{self, Name},
    MacroDefId,
};
use ra_db::CrateId;
use rustc_hash::FxHashSet;

#[derive(Debug, Clone, Default)]
pub(crate) struct Resolver {
    scopes: Vec<Scope>,
}

// FIXME how to store these best
#[derive(Debug, Clone)]
pub(crate) struct ModuleItemMap {
    crate_def_map: Arc<CrateDefMap>,
    module_id: CrateModuleId,
}

#[derive(Debug, Clone)]
pub(crate) struct ExprScope {
    owner: DefWithBodyId,
    expr_scopes: Arc<ExprScopes>,
    scope_id: ScopeId,
}

#[derive(Debug, Clone)]
pub(crate) enum Scope {
    /// All the items and imported names of a module
    ModuleScope(ModuleItemMap),
    /// Brings the generic parameters of an item into scope
    GenericParams { def: GenericDefId, params: Arc<GenericParams> },
    /// Brings `Self` in `impl` block into scope
    ImplBlockScope(ImplId),
    /// Brings `Self` in enum, struct and union definitions into scope
    AdtScope(AdtId),
    /// Local bindings
    ExprScope(ExprScope),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TypeNs {
    SelfType(ImplId),
    GenericParam(u32),
    AdtId(AdtId),
    AdtSelfType(AdtId),
    EnumVariantId(EnumVariantId),
    TypeAliasId(TypeAliasId),
    BuiltinType(BuiltinType),
    TraitId(TraitId),
    // Module belong to type ns, but the resolver is used when all module paths
    // are fully resolved.
    // ModuleId(ModuleId)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ResolveValueResult {
    ValueNs(ValueNs),
    Partial(TypeNs, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ValueNs {
    LocalBinding(PatId),
    FunctionId(FunctionId),
    ConstId(ConstId),
    StaticId(StaticId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
}

impl Resolver {
    /// Resolve known trait from std, like `std::futures::Future`
    pub(crate) fn resolve_known_trait(
        &self,
        db: &impl DefDatabase2,
        path: &Path,
    ) -> Option<TraitId> {
        let res = self.resolve_module_path(db, path).take_types()?;
        match res {
            ModuleDefId::TraitId(it) => Some(it),
            _ => None,
        }
    }

    /// Resolve known struct from std, like `std::boxed::Box`
    pub(crate) fn resolve_known_struct(
        &self,
        db: &impl DefDatabase2,
        path: &Path,
    ) -> Option<StructId> {
        let res = self.resolve_module_path(db, path).take_types()?;
        match res {
            ModuleDefId::AdtId(AdtId::StructId(it)) => Some(it),
            _ => None,
        }
    }

    /// Resolve known enum from std, like `std::result::Result`
    pub(crate) fn resolve_known_enum(&self, db: &impl DefDatabase2, path: &Path) -> Option<EnumId> {
        let res = self.resolve_module_path(db, path).take_types()?;
        match res {
            ModuleDefId::AdtId(AdtId::EnumId(it)) => Some(it),
            _ => None,
        }
    }

    /// pub only for source-binder
    pub(crate) fn resolve_module_path(&self, db: &impl DefDatabase2, path: &Path) -> PerNs {
        let (item_map, module) = match self.module() {
            Some(it) => it,
            None => return PerNs::none(),
        };
        let (module_res, segment_index) = item_map.resolve_path(db, module, path);
        if segment_index.is_some() {
            return PerNs::none();
        }
        module_res
    }

    pub(crate) fn resolve_path_in_type_ns(
        &self,
        db: &impl DefDatabase2,
        path: &Path,
    ) -> Option<(TypeNs, Option<usize>)> {
        if path.is_type_relative() {
            return None;
        }
        let first_name = &path.segments.first()?.name;
        let skip_to_mod = path.kind != PathKind::Plain;
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::ExprScope(_) => continue,
                Scope::GenericParams { .. } | Scope::ImplBlockScope(_) if skip_to_mod => continue,

                Scope::GenericParams { params, .. } => {
                    if let Some(param) = params.find_by_name(first_name) {
                        let idx = if path.segments.len() == 1 { None } else { Some(1) };
                        return Some((TypeNs::GenericParam(param.idx), idx));
                    }
                }
                Scope::ImplBlockScope(impl_) => {
                    if first_name == &name::SELF_TYPE {
                        let idx = if path.segments.len() == 1 { None } else { Some(1) };
                        return Some((TypeNs::SelfType(*impl_), idx));
                    }
                }
                Scope::AdtScope(adt) => {
                    if first_name == &name::SELF_TYPE {
                        let idx = if path.segments.len() == 1 { None } else { Some(1) };
                        return Some((TypeNs::AdtSelfType(*adt), idx));
                    }
                }
                Scope::ModuleScope(m) => {
                    let (module_def, idx) = m.crate_def_map.resolve_path(db, m.module_id, path);
                    let res = match module_def.take_types()? {
                        ModuleDefId::AdtId(it) => TypeNs::AdtId(it),
                        ModuleDefId::EnumVariantId(it) => TypeNs::EnumVariantId(it),

                        ModuleDefId::TypeAliasId(it) => TypeNs::TypeAliasId(it),
                        ModuleDefId::BuiltinType(it) => TypeNs::BuiltinType(it),

                        ModuleDefId::TraitId(it) => TypeNs::TraitId(it),

                        ModuleDefId::FunctionId(_)
                        | ModuleDefId::ConstId(_)
                        | ModuleDefId::StaticId(_)
                        | ModuleDefId::ModuleId(_) => return None,
                    };
                    return Some((res, idx));
                }
            }
        }
        None
    }

    pub(crate) fn resolve_path_in_type_ns_fully(
        &self,
        db: &impl DefDatabase2,
        path: &Path,
    ) -> Option<TypeNs> {
        let (res, unresolved) = self.resolve_path_in_type_ns(db, path)?;
        if unresolved.is_some() {
            return None;
        }
        Some(res)
    }

    pub(crate) fn resolve_path_in_value_ns<'p>(
        &self,
        db: &impl DefDatabase2,
        path: &'p Path,
    ) -> Option<ResolveValueResult> {
        if path.is_type_relative() {
            return None;
        }
        let n_segments = path.segments.len();
        let tmp = name::SELF_PARAM;
        let first_name = if path.is_self() { &tmp } else { &path.segments.first()?.name };
        let skip_to_mod = path.kind != PathKind::Plain && !path.is_self();
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::AdtScope(_)
                | Scope::ExprScope(_)
                | Scope::GenericParams { .. }
                | Scope::ImplBlockScope(_)
                    if skip_to_mod =>
                {
                    continue
                }

                Scope::ExprScope(scope) if n_segments <= 1 => {
                    let entry = scope
                        .expr_scopes
                        .entries(scope.scope_id)
                        .iter()
                        .find(|entry| entry.name() == first_name);

                    if let Some(e) = entry {
                        return Some(ResolveValueResult::ValueNs(ValueNs::LocalBinding(e.pat())));
                    }
                }
                Scope::ExprScope(_) => continue,

                Scope::GenericParams { params, .. } if n_segments > 1 => {
                    if let Some(param) = params.find_by_name(first_name) {
                        let ty = TypeNs::GenericParam(param.idx);
                        return Some(ResolveValueResult::Partial(ty, 1));
                    }
                }
                Scope::GenericParams { .. } => continue,

                Scope::ImplBlockScope(impl_) if n_segments > 1 => {
                    if first_name == &name::SELF_TYPE {
                        let ty = TypeNs::SelfType(*impl_);
                        return Some(ResolveValueResult::Partial(ty, 1));
                    }
                }
                Scope::AdtScope(adt) if n_segments > 1 => {
                    if first_name == &name::SELF_TYPE {
                        let ty = TypeNs::AdtSelfType(*adt);
                        return Some(ResolveValueResult::Partial(ty, 1));
                    }
                }
                Scope::ImplBlockScope(_) | Scope::AdtScope(_) => continue,

                Scope::ModuleScope(m) => {
                    let (module_def, idx) = m.crate_def_map.resolve_path(db, m.module_id, path);
                    return match idx {
                        None => {
                            let value = match module_def.take_values()? {
                                ModuleDefId::FunctionId(it) => ValueNs::FunctionId(it),
                                ModuleDefId::AdtId(AdtId::StructId(it)) => ValueNs::StructId(it),
                                ModuleDefId::EnumVariantId(it) => ValueNs::EnumVariantId(it),
                                ModuleDefId::ConstId(it) => ValueNs::ConstId(it),
                                ModuleDefId::StaticId(it) => ValueNs::StaticId(it),

                                ModuleDefId::AdtId(AdtId::EnumId(_))
                                | ModuleDefId::AdtId(AdtId::UnionId(_))
                                | ModuleDefId::TraitId(_)
                                | ModuleDefId::TypeAliasId(_)
                                | ModuleDefId::BuiltinType(_)
                                | ModuleDefId::ModuleId(_) => return None,
                            };
                            Some(ResolveValueResult::ValueNs(value))
                        }
                        Some(idx) => {
                            let ty = match module_def.take_types()? {
                                ModuleDefId::AdtId(it) => TypeNs::AdtId(it),
                                ModuleDefId::TraitId(it) => TypeNs::TraitId(it),
                                ModuleDefId::TypeAliasId(it) => TypeNs::TypeAliasId(it),
                                ModuleDefId::BuiltinType(it) => TypeNs::BuiltinType(it),

                                ModuleDefId::ModuleId(_)
                                | ModuleDefId::FunctionId(_)
                                | ModuleDefId::EnumVariantId(_)
                                | ModuleDefId::ConstId(_)
                                | ModuleDefId::StaticId(_) => return None,
                            };
                            Some(ResolveValueResult::Partial(ty, idx))
                        }
                    };
                }
            }
        }
        None
    }

    pub(crate) fn resolve_path_in_value_ns_fully(
        &self,
        db: &impl DefDatabase2,
        path: &Path,
    ) -> Option<ValueNs> {
        match self.resolve_path_in_value_ns(db, path)? {
            ResolveValueResult::ValueNs(it) => Some(it),
            ResolveValueResult::Partial(..) => None,
        }
    }

    pub(crate) fn resolve_path_as_macro(
        &self,
        db: &impl DefDatabase2,
        path: &Path,
    ) -> Option<MacroDefId> {
        let (item_map, module) = self.module()?;
        item_map.resolve_path(db, module, path).0.get_macros()
    }

    pub(crate) fn process_all_names(
        &self,
        db: &impl DefDatabase2,
        f: &mut dyn FnMut(Name, ScopeDef),
    ) {
        for scope in self.scopes.iter().rev() {
            scope.process_names(db, f);
        }
    }

    pub(crate) fn traits_in_scope(&self, db: &impl DefDatabase2) -> FxHashSet<TraitId> {
        let mut traits = FxHashSet::default();
        for scope in &self.scopes {
            if let Scope::ModuleScope(m) = scope {
                if let Some(prelude) = m.crate_def_map.prelude() {
                    let prelude_def_map = db.crate_def_map(prelude.krate);
                    traits.extend(prelude_def_map[prelude.module_id].scope.traits());
                }
                traits.extend(m.crate_def_map[m.module_id].scope.traits());
            }
        }
        traits
    }

    fn module(&self) -> Option<(&CrateDefMap, CrateModuleId)> {
        self.scopes.iter().rev().find_map(|scope| match scope {
            Scope::ModuleScope(m) => Some((&*m.crate_def_map, m.module_id)),

            _ => None,
        })
    }

    pub(crate) fn krate(&self) -> Option<CrateId> {
        self.module().map(|t| t.0.krate())
    }

    pub(crate) fn where_predicates_in_scope<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'a crate::generics::WherePredicate> + 'a {
        self.scopes
            .iter()
            .filter_map(|scope| match scope {
                Scope::GenericParams { params, .. } => Some(params),
                _ => None,
            })
            .flat_map(|params| params.where_predicates.iter())
    }

    pub(crate) fn generic_def(&self) -> Option<GenericDefId> {
        self.scopes.iter().find_map(|scope| match scope {
            Scope::GenericParams { def, .. } => Some(*def),
            _ => None,
        })
    }

    pub(crate) fn body_owner(&self) -> Option<DefWithBodyId> {
        self.scopes.iter().find_map(|scope| match scope {
            Scope::ExprScope(it) => Some(it.owner),
            _ => None,
        })
    }
}

impl Resolver {
    pub(crate) fn push_scope(mut self, scope: Scope) -> Resolver {
        self.scopes.push(scope);
        self
    }

    pub(crate) fn push_generic_params_scope(
        self,
        db: &impl DefDatabase2,
        def: GenericDefId,
    ) -> Resolver {
        let params = db.generic_params(def);
        if params.params.is_empty() {
            self
        } else {
            self.push_scope(Scope::GenericParams { def, params })
        }
    }

    pub(crate) fn push_impl_block_scope(self, impl_block: ImplId) -> Resolver {
        self.push_scope(Scope::ImplBlockScope(impl_block))
    }

    pub(crate) fn push_module_scope(
        self,
        crate_def_map: Arc<CrateDefMap>,
        module_id: CrateModuleId,
    ) -> Resolver {
        self.push_scope(Scope::ModuleScope(ModuleItemMap { crate_def_map, module_id }))
    }

    pub(crate) fn push_expr_scope(
        self,
        owner: DefWithBodyId,
        expr_scopes: Arc<ExprScopes>,
        scope_id: ScopeId,
    ) -> Resolver {
        self.push_scope(Scope::ExprScope(ExprScope { owner, expr_scopes, scope_id }))
    }
}

pub(crate) enum ScopeDef {
    PerNs(PerNs),
    ImplSelfType(ImplId),
    AdtSelfType(AdtId),
    GenericParam(u32),
    Local(PatId),
}

impl Scope {
    fn process_names(&self, db: &impl DefDatabase2, f: &mut dyn FnMut(Name, ScopeDef)) {
        match self {
            Scope::ModuleScope(m) => {
                // FIXME: should we provide `self` here?
                // f(
                //     Name::self_param(),
                //     PerNs::types(Resolution::Def {
                //         def: m.module.into(),
                //     }),
                // );
                m.crate_def_map[m.module_id].scope.entries().for_each(|(name, res)| {
                    f(name.clone(), ScopeDef::PerNs(res.def));
                });
                m.crate_def_map[m.module_id].scope.legacy_macros().for_each(|(name, macro_)| {
                    f(name.clone(), ScopeDef::PerNs(PerNs::macros(macro_)));
                });
                m.crate_def_map.extern_prelude().iter().for_each(|(name, &def)| {
                    f(name.clone(), ScopeDef::PerNs(PerNs::types(def.into())));
                });
                if let Some(prelude) = m.crate_def_map.prelude() {
                    let prelude_def_map = db.crate_def_map(prelude.krate);
                    prelude_def_map[prelude.module_id].scope.entries().for_each(|(name, res)| {
                        f(name.clone(), ScopeDef::PerNs(res.def));
                    });
                }
            }
            Scope::GenericParams { params, .. } => {
                for param in params.params.iter() {
                    f(param.name.clone(), ScopeDef::GenericParam(param.idx))
                }
            }
            Scope::ImplBlockScope(i) => {
                f(name::SELF_TYPE, ScopeDef::ImplSelfType((*i).into()));
            }
            Scope::AdtScope(i) => {
                f(name::SELF_TYPE, ScopeDef::AdtSelfType((*i).into()));
            }
            Scope::ExprScope(scope) => {
                scope.expr_scopes.entries(scope.scope_id).iter().for_each(|e| {
                    f(e.name().clone(), ScopeDef::Local(e.pat()));
                });
            }
        }
    }
}

// needs arbitrary_self_types to be a method... or maybe move to the def?
pub(crate) fn resolver_for_expr(
    db: &impl DefDatabase2,
    owner: DefWithBodyId,
    expr_id: ExprId,
) -> Resolver {
    let scopes = db.expr_scopes(owner);
    resolver_for_scope(db, owner, scopes.scope_for(expr_id))
}

pub(crate) fn resolver_for_scope(
    db: &impl DefDatabase2,
    owner: DefWithBodyId,
    scope_id: Option<ScopeId>,
) -> Resolver {
    let mut r = owner.resolver(db);
    let scopes = db.expr_scopes(owner);
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    for scope in scope_chain.into_iter().rev() {
        r = r.push_expr_scope(owner, Arc::clone(&scopes), scope);
    }
    r
}

pub(crate) trait HasResolver {
    /// Builds a resolver for type references inside this def.
    fn resolver(self, db: &impl DefDatabase2) -> Resolver;
}

impl HasResolver for ModuleId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        let def_map = db.crate_def_map(self.krate);
        Resolver::default().push_module_scope(def_map, self.module_id)
    }
}

impl HasResolver for TraitId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        self.module(db).resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for AdtId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        let module = match self {
            AdtId::StructId(it) => it.0.module(db),
            AdtId::UnionId(it) => it.0.module(db),
            AdtId::EnumId(it) => it.module(db),
        };

        module
            .resolver(db)
            .push_generic_params_scope(db, self.into())
            .push_scope(Scope::AdtScope(self.into()))
    }
}

impl HasResolver for StructId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        AdtId::from(self).resolver(db)
    }
}

impl HasResolver for UnionId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        AdtId::from(self).resolver(db)
    }
}

impl HasResolver for EnumId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        AdtId::from(self).resolver(db)
    }
}

impl HasResolver for FunctionId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        self.lookup(db).container.resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for DefWithBodyId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        match self {
            DefWithBodyId::ConstId(c) => c.resolver(db),
            DefWithBodyId::FunctionId(f) => f.resolver(db),
            DefWithBodyId::StaticId(s) => s.resolver(db),
        }
    }
}

impl HasResolver for ConstId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        self.lookup(db).container.resolver(db)
    }
}

impl HasResolver for StaticId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        self.module(db).resolver(db)
    }
}

impl HasResolver for TypeAliasId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        self.lookup(db).container.resolver(db).push_generic_params_scope(db, self.into())
    }
}

impl HasResolver for ContainerId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        match self {
            ContainerId::TraitId(it) => it.resolver(db),
            ContainerId::ImplId(it) => it.resolver(db),
            ContainerId::ModuleId(it) => it.resolver(db),
        }
    }
}

impl HasResolver for GenericDefId {
    fn resolver(self, db: &impl DefDatabase2) -> crate::Resolver {
        match self {
            GenericDefId::FunctionId(inner) => inner.resolver(db),
            GenericDefId::AdtId(adt) => adt.resolver(db),
            GenericDefId::TraitId(inner) => inner.resolver(db),
            GenericDefId::TypeAliasId(inner) => inner.resolver(db),
            GenericDefId::ImplId(inner) => inner.resolver(db),
            GenericDefId::EnumVariantId(inner) => inner.parent.resolver(db),
            GenericDefId::ConstId(inner) => inner.resolver(db),
        }
    }
}

impl HasResolver for ImplId {
    fn resolver(self, db: &impl DefDatabase2) -> Resolver {
        self.module(db)
            .resolver(db)
            .push_generic_params_scope(db, self.into())
            .push_impl_block_scope(self)
    }
}
