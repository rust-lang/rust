#![allow(unused_variables, dead_code)]
//! Name resolution.
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::{
    ModuleDef,
    name::Name,
    nameres::{PerNs, lower::ImportId, ItemMap},
    module_tree::ModuleId,
    generics::GenericParams,
    expr::{Body, scope::{ExprScopes, ScopeId}, PatId},
    impl_block::ImplBlock,
    path::Path,
};

#[derive(Debug, Clone, Default)]
pub struct Resolver {
    scopes: Vec<Scope>, // maybe a 'linked list' of scopes? or allow linking a Resolver to a parent Resolver? that's an optimization that might not be necessary, though
}

// TODO how to store these best
#[derive(Debug, Clone)]
pub(crate) struct ModuleItemMap {
    item_map: Arc<ItemMap>,
    module_id: ModuleId,
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
    /// Brings the function parameters into scope
    FunctionParams(Arc<Body>),
    /// Brings `Self` into scope
    ImplBlockScope(ImplBlock),
    /// Local bindings
    ExprScope(ExprScope),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// An item
    Def {
        def: ModuleDef,
        import: Option<ImportId>,
    },
    /// A local binding (only value namespace)
    LocalBinding { pat: PatId },
    /// A generic parameter
    GenericParam { idx: u32 },
    // TODO how does `Self` resolve?
}

impl Resolver {
    pub fn resolve_name(&self, name: &Name) -> PerNs<Resolution> {
        for scope in self.scopes.iter().rev() {
            let resolution = scope.resolve_name(name);
            if !resolution.is_none() {
                return resolution;
            }
        }
        PerNs::none()
    }

    pub fn resolve_path(&self, path: &Path) -> PerNs<Resolution> {
        unimplemented!()
    }

    pub fn all_names(&self) -> FxHashMap<Name, Resolution> {
        unimplemented!()
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

    pub(crate) fn push_module_scope(self, item_map: Arc<ItemMap>, module_id: ModuleId) -> Resolver {
        self.push_scope(Scope::ModuleScope(ModuleItemMap {
            item_map,
            module_id,
        }))
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

    pub(crate) fn push_function_params(self, body: Arc<Body>) -> Resolver {
        self.push_scope(Scope::FunctionParams(body))
    }

    pub(crate) fn pop_scope(mut self) -> Resolver {
        self.scopes.pop();
        self
    }
}

impl Scope {
    fn resolve_name(&self, name: &Name) -> PerNs<Resolution> {
        unimplemented!()
    }
}
