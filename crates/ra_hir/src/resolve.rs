//! Name resolution.
use std::sync::Arc;

use hir_def::{
    builtin_type::BuiltinType,
    nameres::CrateDefMap,
    path::{Path, PathKind},
    AdtId, CrateModuleId, ModuleDefId,
};
use hir_expand::name::{self, Name};
use rustc_hash::FxHashSet;

use crate::{
    code_model::Crate,
    db::{DefDatabase, HirDatabase},
    expr::{
        scope::{ExprScopes, ScopeId},
        PatId,
    },
    generics::GenericParams,
    impl_block::ImplBlock,
    Adt, Const, Enum, EnumVariant, Function, MacroDef, ModuleDef, PerNs, Static, Struct, Trait,
    TypeAlias,
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
    expr_scopes: Arc<ExprScopes>,
    scope_id: ScopeId,
}

#[derive(Debug, Clone)]
pub(crate) enum Scope {
    /// All the items and imported names of a module
    ModuleScope(ModuleItemMap),
    /// Brings the generic parameters of an item into scope
    GenericParams(Arc<GenericParams>),
    /// Brings `Self` in `impl` block into scope
    ImplBlockScope(ImplBlock),
    /// Brings `Self` in enum, struct and union definitions into scope
    AdtScope(Adt),
    /// Local bindings
    ExprScope(ExprScope),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeNs {
    SelfType(ImplBlock),
    GenericParam(u32),
    Adt(Adt),
    AdtSelfType(Adt),
    EnumVariant(EnumVariant),
    TypeAlias(TypeAlias),
    BuiltinType(BuiltinType),
    Trait(Trait),
    // Module belong to type ns, but the resolver is used when all module paths
    // are fully resolved.
    // Module(Module)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResolveValueResult {
    ValueNs(ValueNs),
    Partial(TypeNs, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueNs {
    LocalBinding(PatId),
    Function(Function),
    Const(Const),
    Static(Static),
    Struct(Struct),
    EnumVariant(EnumVariant),
}

impl Resolver {
    /// Resolve known trait from std, like `std::futures::Future`
    pub(crate) fn resolve_known_trait(&self, db: &impl HirDatabase, path: &Path) -> Option<Trait> {
        let res = self.resolve_module_path(db, path).take_types()?;
        match res {
            ModuleDefId::TraitId(it) => Some(it.into()),
            _ => None,
        }
    }

    /// Resolve known struct from std, like `std::boxed::Box`
    pub(crate) fn resolve_known_struct(
        &self,
        db: &impl HirDatabase,
        path: &Path,
    ) -> Option<Struct> {
        let res = self.resolve_module_path(db, path).take_types()?;
        match res {
            ModuleDefId::AdtId(AdtId::StructId(it)) => Some(it.into()),
            _ => None,
        }
    }

    /// Resolve known enum from std, like `std::result::Result`
    pub(crate) fn resolve_known_enum(&self, db: &impl HirDatabase, path: &Path) -> Option<Enum> {
        let res = self.resolve_module_path(db, path).take_types()?;
        match res {
            ModuleDefId::AdtId(AdtId::EnumId(it)) => Some(it.into()),
            _ => None,
        }
    }

    /// pub only for source-binder
    pub(crate) fn resolve_module_path(&self, db: &impl HirDatabase, path: &Path) -> PerNs {
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
        db: &impl HirDatabase,
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
                Scope::GenericParams(_) | Scope::ImplBlockScope(_) if skip_to_mod => continue,

                Scope::GenericParams(params) => {
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
                        ModuleDefId::AdtId(it) => TypeNs::Adt(it.into()),
                        ModuleDefId::EnumVariantId(it) => TypeNs::EnumVariant(it.into()),

                        ModuleDefId::TypeAliasId(it) => TypeNs::TypeAlias(it.into()),
                        ModuleDefId::BuiltinType(it) => TypeNs::BuiltinType(it),

                        ModuleDefId::TraitId(it) => TypeNs::Trait(it.into()),

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
        db: &impl HirDatabase,
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
        db: &impl HirDatabase,
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
                | Scope::GenericParams(_)
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

                Scope::GenericParams(params) if n_segments > 1 => {
                    if let Some(param) = params.find_by_name(first_name) {
                        let ty = TypeNs::GenericParam(param.idx);
                        return Some(ResolveValueResult::Partial(ty, 1));
                    }
                }
                Scope::GenericParams(_) => continue,

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
                                ModuleDefId::AdtId(it) => TypeNs::Adt(it.into()),
                                ModuleDefId::TraitId(it) => TypeNs::Trait(it.into()),
                                ModuleDefId::TypeAliasId(it) => TypeNs::TypeAlias(it.into()),
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
        db: &impl HirDatabase,
        path: &Path,
    ) -> Option<ValueNs> {
        match self.resolve_path_in_value_ns(db, path)? {
            ResolveValueResult::ValueNs(it) => Some(it),
            ResolveValueResult::Partial(..) => None,
        }
    }

    pub(crate) fn resolve_path_as_macro(
        &self,
        db: &impl DefDatabase,
        path: &Path,
    ) -> Option<MacroDef> {
        let (item_map, module) = self.module()?;
        item_map.resolve_path(db, module, path).0.get_macros().map(MacroDef::from)
    }

    pub(crate) fn process_all_names(
        &self,
        db: &impl HirDatabase,
        f: &mut dyn FnMut(Name, ScopeDef),
    ) {
        for scope in self.scopes.iter().rev() {
            scope.process_names(db, f);
        }
    }

    pub(crate) fn traits_in_scope(&self, db: &impl HirDatabase) -> FxHashSet<Trait> {
        let mut traits = FxHashSet::default();
        for scope in &self.scopes {
            if let Scope::ModuleScope(m) = scope {
                if let Some(prelude) = m.crate_def_map.prelude() {
                    let prelude_def_map = db.crate_def_map(prelude.krate);
                    traits
                        .extend(prelude_def_map[prelude.module_id].scope.traits().map(Trait::from));
                }
                traits.extend(m.crate_def_map[m.module_id].scope.traits().map(Trait::from));
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
                Scope::GenericParams(params) => Some(params),
                _ => None,
            })
            .flat_map(|params| params.where_predicates.iter())
    }

    pub(crate) fn generic_def(&self) -> Option<crate::generics::GenericDef> {
        self.scopes.iter().find_map(|scope| match scope {
            Scope::GenericParams(params) => Some(params.def),
            _ => None,
        })
    }
}

impl Resolver {
    pub(crate) fn push_scope(mut self, scope: Scope) -> Resolver {
        self.scopes.push(scope);
        self
    }

    pub(crate) fn push_generic_params_scope(self, params: Arc<GenericParams>) -> Resolver {
        self.push_scope(Scope::GenericParams(params))
    }

    pub(crate) fn push_impl_block_scope(self, impl_block: ImplBlock) -> Resolver {
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
        expr_scopes: Arc<ExprScopes>,
        scope_id: ScopeId,
    ) -> Resolver {
        self.push_scope(Scope::ExprScope(ExprScope { expr_scopes, scope_id }))
    }
}

/// For IDE only
pub enum ScopeDef {
    ModuleDef(ModuleDef),
    MacroDef(MacroDef),
    GenericParam(u32),
    ImplSelfType(ImplBlock),
    AdtSelfType(Adt),
    LocalBinding(PatId),
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
    fn process_names(&self, db: &impl HirDatabase, f: &mut dyn FnMut(Name, ScopeDef)) {
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
            Scope::GenericParams(gp) => {
                for param in &gp.params {
                    f(param.name.clone(), ScopeDef::GenericParam(param.idx))
                }
            }
            Scope::ImplBlockScope(i) => {
                f(name::SELF_TYPE, ScopeDef::ImplSelfType(*i));
            }
            Scope::AdtScope(i) => {
                f(name::SELF_TYPE, ScopeDef::AdtSelfType(*i));
            }
            Scope::ExprScope(e) => {
                e.expr_scopes.entries(e.scope_id).iter().for_each(|e| {
                    f(e.name().clone(), ScopeDef::LocalBinding(e.pat()));
                });
            }
        }
    }
}
