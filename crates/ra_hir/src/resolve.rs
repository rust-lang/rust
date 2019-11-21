//! Name resolution.
use std::sync::Arc;

use hir_def::{
    builtin_type::BuiltinType,
    db::DefDatabase2,
    generics::GenericParams,
    nameres::CrateDefMap,
    path::{Path, PathKind},
    AdtId, CrateModuleId, DefWithBodyId, EnumId, EnumVariantId, GenericDefId, ImplId, ModuleDefId,
    StructId, TraitId, TypeAliasId,
};
use hir_expand::name::{self, Name};
use rustc_hash::FxHashSet;

use crate::{
    code_model::Crate,
    db::DefDatabase,
    expr::{ExprScopes, PatId, ScopeId},
    Adt, Const, Container, DefWithBody, EnumVariant, Function, GenericDef, ImplBlock, Local,
    MacroDef, Module, ModuleDef, PerNs, Static, Struct, Trait, TypeAlias,
};

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
    Function(Function),
    Const(Const),
    Static(Static),
    Struct(Struct),
    EnumVariant(EnumVariant),
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
                                ModuleDefId::FunctionId(it) => ValueNs::Function(it.into()),
                                ModuleDefId::AdtId(AdtId::StructId(it)) => {
                                    ValueNs::Struct(it.into())
                                }
                                ModuleDefId::EnumVariantId(it) => ValueNs::EnumVariant(it.into()),
                                ModuleDefId::ConstId(it) => ValueNs::Const(it.into()),
                                ModuleDefId::StaticId(it) => ValueNs::Static(it.into()),

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
    ) -> Option<MacroDef> {
        let (item_map, module) = self.module()?;
        item_map.resolve_path(db, module, path).0.get_macros().map(MacroDef::from)
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

    pub(crate) fn krate(&self) -> Option<Crate> {
        self.module().map(|t| Crate { crate_id: t.0.krate() })
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

/// For IDE only
pub enum ScopeDef {
    ModuleDef(ModuleDef),
    MacroDef(MacroDef),
    GenericParam(u32),
    ImplSelfType(ImplBlock),
    AdtSelfType(Adt),
    Local(Local),
    Unknown,
}

impl From<PerNs> for ScopeDef {
    fn from(def: PerNs) -> Self {
        def.take_types()
            .or_else(|| def.take_values())
            .map(|module_def_id| ScopeDef::ModuleDef(module_def_id.into()))
            .or_else(|| {
                def.get_macros().map(|macro_def_id| ScopeDef::MacroDef(macro_def_id.into()))
            })
            .unwrap_or(ScopeDef::Unknown)
    }
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
                    f(name.clone(), res.def.into());
                });
                m.crate_def_map[m.module_id].scope.legacy_macros().for_each(|(name, macro_)| {
                    f(name.clone(), ScopeDef::MacroDef(macro_.into()));
                });
                m.crate_def_map.extern_prelude().iter().for_each(|(name, &def)| {
                    f(name.clone(), ScopeDef::ModuleDef(def.into()));
                });
                if let Some(prelude) = m.crate_def_map.prelude() {
                    let prelude_def_map = db.crate_def_map(prelude.krate);
                    prelude_def_map[prelude.module_id].scope.entries().for_each(|(name, res)| {
                        f(name.clone(), res.def.into());
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
                    let local = Local { parent: scope.owner.into(), pat_id: e.pat() };
                    f(e.name().clone(), ScopeDef::Local(local));
                });
            }
        }
    }
}

pub(crate) trait HasResolver {
    /// Builds a resolver for type references inside this def.
    fn resolver(self, db: &impl DefDatabase) -> Resolver;
}

impl HasResolver for Module {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        let def_map = db.crate_def_map(self.id.krate);
        Resolver::default().push_module_scope(def_map, self.id.module_id)
    }
}

impl HasResolver for Trait {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        self.module(db).resolver(db).push_generic_params_scope(db, self.id.into())
    }
}

impl<T: Into<Adt>> HasResolver for T {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        let def = self.into();
        def.module(db)
            .resolver(db)
            .push_generic_params_scope(db, def.into())
            .push_scope(Scope::AdtScope(def.into()))
    }
}

impl HasResolver for Function {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        self.container(db)
            .map(|c| c.resolver(db))
            .unwrap_or_else(|| self.module(db).resolver(db))
            .push_generic_params_scope(db, self.id.into())
    }
}

impl HasResolver for DefWithBody {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        match self {
            DefWithBody::Const(c) => c.resolver(db),
            DefWithBody::Function(f) => f.resolver(db),
            DefWithBody::Static(s) => s.resolver(db),
        }
    }
}

impl HasResolver for Const {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        self.container(db).map(|c| c.resolver(db)).unwrap_or_else(|| self.module(db).resolver(db))
    }
}

impl HasResolver for Static {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        self.module(db).resolver(db)
    }
}

impl HasResolver for TypeAlias {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        self.container(db)
            .map(|ib| ib.resolver(db))
            .unwrap_or_else(|| self.module(db).resolver(db))
            .push_generic_params_scope(db, self.id.into())
    }
}

impl HasResolver for Container {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        match self {
            Container::Trait(trait_) => trait_.resolver(db),
            Container::ImplBlock(impl_block) => impl_block.resolver(db),
        }
    }
}

impl HasResolver for GenericDef {
    fn resolver(self, db: &impl DefDatabase) -> crate::Resolver {
        match self {
            GenericDef::Function(inner) => inner.resolver(db),
            GenericDef::Adt(adt) => adt.resolver(db),
            GenericDef::Trait(inner) => inner.resolver(db),
            GenericDef::TypeAlias(inner) => inner.resolver(db),
            GenericDef::ImplBlock(inner) => inner.resolver(db),
            GenericDef::EnumVariant(inner) => inner.parent_enum(db).resolver(db),
            GenericDef::Const(inner) => inner.resolver(db),
        }
    }
}

impl HasResolver for ImplBlock {
    fn resolver(self, db: &impl DefDatabase) -> Resolver {
        self.module(db)
            .resolver(db)
            .push_generic_params_scope(db, self.id.into())
            .push_impl_block_scope(self.id)
    }
}
