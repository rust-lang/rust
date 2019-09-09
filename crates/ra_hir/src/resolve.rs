//! Name resolution.
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    code_model::Crate,
    db::HirDatabase,
    expr::{
        scope::{ExprScopes, ScopeId},
        PatId,
    },
    generics::GenericParams,
    impl_block::ImplBlock,
    name::{Name, SELF_PARAM, SELF_TYPE},
    nameres::{CrateDefMap, CrateModuleId, PerNs},
    path::Path,
    MacroDef, ModuleDef, Trait,
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
pub(crate) struct PathResult {
    /// The actual path resolution
    resolution: PerNs<Resolution>,
    /// The first index in the path that we
    /// were unable to resolve.
    /// When path is fully resolved, this is 0.
    remaining_index: usize,
}

impl PathResult {
    /// Returns the remaining index in the result
    /// returns None if the path was fully resolved
    pub(crate) fn remaining_index(&self) -> Option<usize> {
        if self.remaining_index > 0 {
            Some(self.remaining_index)
        } else {
            None
        }
    }

    /// Consumes `PathResult` and returns the contained `PerNs<Resolution>`
    /// if the path was fully resolved, meaning we have no remaining items
    pub(crate) fn into_fully_resolved(self) -> PerNs<Resolution> {
        if self.is_fully_resolved() {
            self.resolution
        } else {
            PerNs::none()
        }
    }

    /// Consumes `PathResult` and returns the resolution and the
    /// remaining_index as a tuple.
    pub(crate) fn into_inner(self) -> (PerNs<Resolution>, Option<usize>) {
        let index = self.remaining_index();
        (self.resolution, index)
    }

    /// Path is fully resolved when `remaining_index` is none
    /// and the resolution contains anything
    pub(crate) fn is_fully_resolved(&self) -> bool {
        !self.resolution.is_none() && self.remaining_index().is_none()
    }

    fn empty() -> PathResult {
        PathResult { resolution: PerNs::none(), remaining_index: 0 }
    }

    fn from_resolution(res: PerNs<Resolution>) -> PathResult {
        PathResult::from_resolution_with_index(res, 0)
    }

    fn from_resolution_with_index(res: PerNs<Resolution>, remaining_index: usize) -> PathResult {
        if res.is_none() {
            PathResult::empty()
        } else {
            PathResult { resolution: res, remaining_index }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Scope {
    /// All the items and imported names of a module
    ModuleScope(ModuleItemMap),
    /// Brings the generic parameters of an item into scope
    GenericParams(Arc<GenericParams>),
    /// Brings `Self` into scope
    ImplBlockScope(ImplBlock),
    /// Local bindings
    ExprScope(ExprScope),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// An item
    Def(ModuleDef),
    /// A local binding (only value namespace)
    LocalBinding(PatId),
    /// A generic parameter
    GenericParam(u32),
    SelfType(ImplBlock),
}

impl Resolver {
    pub(crate) fn resolve_name(&self, db: &impl HirDatabase, name: &Name) -> PerNs<Resolution> {
        let mut resolution = PerNs::none();
        for scope in self.scopes.iter().rev() {
            resolution = resolution.or(scope.resolve_name(db, name));
            if resolution.is_all() {
                return resolution;
            }
        }
        resolution
    }

    pub(crate) fn resolve_path_as_macro(
        &self,
        db: &impl HirDatabase,
        path: &Path,
    ) -> Option<MacroDef> {
        let (item_map, module) = self.module()?;
        item_map.resolve_path(db, module, path).0.get_macros()
    }

    /// Returns the resolved path segments
    /// Which may be fully resolved, empty or partially resolved.
    pub(crate) fn resolve_path_segments(&self, db: &impl HirDatabase, path: &Path) -> PathResult {
        if let Some(name) = path.as_ident() {
            PathResult::from_resolution(self.resolve_name(db, name))
        } else if path.is_self() {
            PathResult::from_resolution(self.resolve_name(db, &SELF_PARAM))
        } else {
            let (item_map, module) = match self.module() {
                Some(it) => it,
                None => return PathResult::empty(),
            };
            let (module_res, segment_index) = item_map.resolve_path(db, module, path);

            let def = module_res.map(Resolution::Def);

            if let Some(index) = segment_index {
                PathResult::from_resolution_with_index(def, index)
            } else {
                PathResult::from_resolution(def)
            }
        }
    }

    /// Returns the fully resolved path if we were able to resolve it.
    /// otherwise returns `PerNs::none`
    pub(crate) fn resolve_path_without_assoc_items(
        &self,
        db: &impl HirDatabase,
        path: &Path,
    ) -> PerNs<Resolution> {
        // into_fully_resolved() returns the fully resolved path or PerNs::none() otherwise
        self.resolve_path_segments(db, path).into_fully_resolved()
    }

    pub(crate) fn all_names(&self, db: &impl HirDatabase) -> FxHashMap<Name, PerNs<Resolution>> {
        let mut names = FxHashMap::default();
        for scope in self.scopes.iter().rev() {
            scope.collect_names(db, &mut |name, res| {
                let current: &mut PerNs<Resolution> = names.entry(name).or_default();
                if current.types.is_none() {
                    current.types = res.types;
                }
                if current.values.is_none() {
                    current.values = res.values;
                }
                if current.macros.is_none() {
                    current.macros = res.macros;
                }
            });
        }
        names
    }

    pub(crate) fn traits_in_scope(&self, db: &impl HirDatabase) -> FxHashSet<Trait> {
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
        self.module().map(|t| t.0.krate())
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

impl Scope {
    fn resolve_name(&self, db: &impl HirDatabase, name: &Name) -> PerNs<Resolution> {
        match self {
            Scope::ModuleScope(m) => {
                if name == &SELF_PARAM {
                    PerNs::types(Resolution::Def(m.crate_def_map.mk_module(m.module_id).into()))
                } else {
                    m.crate_def_map
                        .resolve_name_in_module(db, m.module_id, name)
                        .map(Resolution::Def)
                }
            }
            Scope::GenericParams(gp) => match gp.find_by_name(name) {
                Some(gp) => PerNs::types(Resolution::GenericParam(gp.idx)),
                None => PerNs::none(),
            },
            Scope::ImplBlockScope(i) => {
                if name == &SELF_TYPE {
                    PerNs::types(Resolution::SelfType(*i))
                } else {
                    PerNs::none()
                }
            }
            Scope::ExprScope(e) => {
                let entry =
                    e.expr_scopes.entries(e.scope_id).iter().find(|entry| entry.name() == name);
                match entry {
                    Some(e) => PerNs::values(Resolution::LocalBinding(e.pat())),
                    None => PerNs::none(),
                }
            }
        }
    }

    fn collect_names(&self, db: &impl HirDatabase, f: &mut dyn FnMut(Name, PerNs<Resolution>)) {
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
                    f(name.clone(), res.def.map(Resolution::Def));
                });
                m.crate_def_map[m.module_id].scope.legacy_macros().for_each(|(name, macro_)| {
                    f(name.clone(), PerNs::macros(macro_));
                });
                m.crate_def_map.extern_prelude().iter().for_each(|(name, def)| {
                    f(name.clone(), PerNs::types(Resolution::Def(*def)));
                });
                if let Some(prelude) = m.crate_def_map.prelude() {
                    let prelude_def_map = db.crate_def_map(prelude.krate);
                    prelude_def_map[prelude.module_id].scope.entries().for_each(|(name, res)| {
                        f(name.clone(), res.def.map(Resolution::Def));
                    });
                }
            }
            Scope::GenericParams(gp) => {
                for param in &gp.params {
                    f(param.name.clone(), PerNs::types(Resolution::GenericParam(param.idx)))
                }
            }
            Scope::ImplBlockScope(i) => {
                f(SELF_TYPE, PerNs::types(Resolution::SelfType(*i)));
            }
            Scope::ExprScope(e) => {
                e.expr_scopes.entries(e.scope_id).iter().for_each(|e| {
                    f(e.name().clone(), PerNs::values(Resolution::LocalBinding(e.pat())));
                });
            }
        }
    }
}
