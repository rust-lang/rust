//! Name resolution.
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    ModuleDef, Module,
    db::HirDatabase,
    name::{Name, KnownName},
    nameres::{PerNs, ItemMap},
    generics::GenericParams,
    expr::{scope::{ExprScopes, ScopeId}, PatId, Body},
    impl_block::ImplBlock,
    path::Path,
};

#[derive(Debug, Clone, Default)]
pub struct Resolver {
    scopes: Vec<Scope>,
}

// TODO how to store these best
#[derive(Debug, Clone)]
pub(crate) struct ModuleItemMap {
    item_map: Arc<ItemMap>,
    module: Module,
}

#[derive(Debug, Clone)]
pub(crate) struct ExprScope {
    expr_scopes: Arc<ExprScopes>,
    scope_id: ScopeId,
}

#[derive(Debug, Clone)]
pub enum PathResult {
    /// Path was fully resolved
    FullyResolved(PerNs<Resolution>),
    /// Path was partially resolved, first element contains the resolution
    /// second contains the index in the Path.segments which we were unable to resolve
    PartiallyResolved(PerNs<Resolution>, usize),
}

impl PathResult {
    pub fn segment_index(&self) -> Option<usize> {
        match self {
            PathResult::FullyResolved(_) => None,
            PathResult::PartiallyResolved(_, ref i) => Some(*i),
        }
    }

    /// Consumes `PathResult` and returns the contained `PerNs<Resolution>`
    pub fn into_per_ns(self) -> PerNs<Resolution> {
        match self {
            PathResult::FullyResolved(def) => def,
            PathResult::PartiallyResolved(def, _) => def,
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
    pub fn resolve_name(&self, db: &impl HirDatabase, name: &Name) -> PerNs<Resolution> {
        let mut resolution = PerNs::none();
        for scope in self.scopes.iter().rev() {
            resolution = resolution.or(scope.resolve_name(db, name));
            if resolution.is_both() {
                return resolution;
            }
        }
        resolution
    }

    pub fn resolve_path(&self, db: &impl HirDatabase, path: &Path) -> PathResult {
        use self::PathResult::*;
        if let Some(name) = path.as_ident() {
            FullyResolved(self.resolve_name(db, name))
        } else if path.is_self() {
            FullyResolved(self.resolve_name(db, &Name::self_param()))
        } else {
            let (item_map, module) = match self.module() {
                Some(m) => m,
                _ => return FullyResolved(PerNs::none()),
            };
            let (module_res, segment_index) = item_map.resolve_path(db, module, path);

            let def = module_res.map(Resolution::Def);

            if let Some(index) = segment_index {
                PartiallyResolved(def, index)
            } else {
                FullyResolved(def)
            }
        }
    }

    pub fn all_names(&self, db: &impl HirDatabase) -> FxHashMap<Name, PerNs<Resolution>> {
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
            });
        }
        names
    }

    fn module(&self) -> Option<(&ItemMap, Module)> {
        self.scopes.iter().rev().find_map(|scope| match scope {
            Scope::ModuleScope(m) => Some((&*m.item_map, m.module.clone())),

            _ => None,
        })
    }

    /// The body from which any `LocalBinding` resolutions in this resolver come.
    pub fn body(&self) -> Option<Arc<Body>> {
        self.scopes.iter().rev().find_map(|scope| match scope {
            Scope::ExprScope(expr_scope) => Some(expr_scope.expr_scopes.body()),
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

    pub(crate) fn push_module_scope(self, item_map: Arc<ItemMap>, module: Module) -> Resolver {
        self.push_scope(Scope::ModuleScope(ModuleItemMap { item_map, module }))
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
                if let Some(KnownName::SelfParam) = name.as_known_name() {
                    PerNs::types(Resolution::Def(m.module.into()))
                } else {
                    m.item_map.resolve_name_in_module(db, m.module, name).map(Resolution::Def)
                }
            }
            Scope::GenericParams(gp) => match gp.find_by_name(name) {
                Some(gp) => PerNs::types(Resolution::GenericParam(gp.idx)),
                None => PerNs::none(),
            },
            Scope::ImplBlockScope(i) => {
                if name.as_known_name() == Some(KnownName::SelfType) {
                    PerNs::types(Resolution::SelfType(i.clone()))
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
                // TODO: should we provide `self` here?
                // f(
                //     Name::self_param(),
                //     PerNs::types(Resolution::Def {
                //         def: m.module.into(),
                //     }),
                // );
                m.item_map[m.module.module_id].entries().for_each(|(name, res)| {
                    f(name.clone(), res.def.map(Resolution::Def));
                });
                m.item_map.extern_prelude.iter().for_each(|(name, def)| {
                    f(name.clone(), PerNs::types(Resolution::Def(*def)));
                });
                if let Some(prelude) = m.item_map.prelude {
                    let prelude_item_map = db.item_map(prelude.krate);
                    prelude_item_map[prelude.module_id].entries().for_each(|(name, res)| {
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
                f(Name::self_type(), PerNs::types(Resolution::SelfType(i.clone())));
            }
            Scope::ExprScope(e) => {
                e.expr_scopes.entries(e.scope_id).iter().for_each(|e| {
                    f(e.name().clone(), PerNs::values(Resolution::LocalBinding(e.pat())));
                });
            }
        }
    }
}
