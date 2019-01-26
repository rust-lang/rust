//! Name resolution.
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    ModuleDef, Module,
    db::HirDatabase,
    name::{Name, KnownName},
    nameres::{PerNs, ItemMap},
    generics::GenericParams,
    expr::{scope::{ExprScopes, ScopeId}, PatId},
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
    // FIXME make these tuple variants
    /// An item
    Def {
        def: ModuleDef,
    },
    /// A local binding (only value namespace)
    LocalBinding {
        pat: PatId,
    },
    /// A generic parameter
    GenericParam {
        idx: u32,
    },
    SelfType(ImplBlock),
}

impl Resolver {
    pub fn resolve_name(&self, name: &Name) -> PerNs<Resolution> {
        let mut resolution = PerNs::none();
        for scope in self.scopes.iter().rev() {
            resolution = resolution.combine(scope.resolve_name(name));
            if resolution.is_both() {
                return resolution;
            }
        }
        resolution
    }

    pub fn resolve_path(&self, db: &impl HirDatabase, path: &Path) -> PerNs<Resolution> {
        if let Some(name) = path.as_ident() {
            self.resolve_name(name)
        } else if path.is_self() {
            self.resolve_name(&Name::self_param())
        } else {
            let (item_map, module) = match self.module() {
                Some(m) => m,
                _ => return PerNs::none(),
            };
            let module_res = item_map.resolve_path(db, module, path);
            module_res.map(|def| Resolution::Def { def })
        }
    }

    pub fn all_names(&self) -> FxHashMap<Name, Resolution> {
        unimplemented!()
    }

    fn module(&self) -> Option<(&ItemMap, Module)> {
        for scope in self.scopes.iter().rev() {
            match scope {
                Scope::ModuleScope(m) => {
                    return Some((&m.item_map, m.module.clone()));
                }
                _ => {}
            }
        }
        None
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
        self.push_scope(Scope::ExprScope(ExprScope {
            expr_scopes,
            scope_id,
        }))
    }
}

impl Scope {
    fn resolve_name(&self, name: &Name) -> PerNs<Resolution> {
        match self {
            Scope::ModuleScope(m) => {
                if let Some(KnownName::SelfParam) = name.as_known_name() {
                    PerNs::types(Resolution::Def {
                        def: m.module.into(),
                    })
                } else {
                    match m.item_map[m.module.module_id].get(name) {
                        Some(res) => res.def.map(|def| Resolution::Def { def }),
                        None => PerNs::none(),
                    }
                }
            }
            Scope::GenericParams(gp) => match gp.find_by_name(name) {
                Some(gp) => PerNs::types(Resolution::GenericParam { idx: gp.idx }),
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
                let entry = e
                    .expr_scopes
                    .entries(e.scope_id)
                    .iter()
                    .find(|entry| entry.name() == name);
                match entry {
                    Some(e) => PerNs::values(Resolution::LocalBinding { pat: e.pat() }),
                    None => PerNs::none(),
                }
            }
        }
    }
}
