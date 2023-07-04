use crate::middle::region::{Scope, ScopeData, ScopeTree};
use rustc_hir as hir;
use rustc_hir::ItemLocalMap;

/// `RvalueScopes` is a mapping from sub-expressions to _extended_ lifetime as determined by
/// rules laid out in `rustc_hir_analysis::check::rvalue_scopes`.
#[derive(TyEncodable, TyDecodable, Clone, Debug, Default, Eq, PartialEq, HashStable)]
pub struct RvalueScopes {
    map: ItemLocalMap<Option<Scope>>,
}

impl RvalueScopes {
    pub fn new() -> Self {
        Self { map: <_>::default() }
    }

    /// Returns the scope when the temp created by `expr_id` will be cleaned up.
    pub fn temporary_scope(
        &self,
        region_scope_tree: &ScopeTree,
        expr_id: hir::ItemLocalId,
    ) -> Option<Scope> {
        // Check for a designated rvalue scope.
        if let Some(&s) = self.map.get(&expr_id) {
            debug!("temporary_scope({expr_id:?}) = {s:?} [custom]");
            return s;
        }

        // Otherwise, locate the innermost terminating scope
        // if there's one. Static items, for instance, won't
        // have an enclosing scope, hence no scope will be
        // returned.
        let mut id = Scope { id: expr_id, data: ScopeData::Node };

        while let Some(&(p, _)) = region_scope_tree.parent_map.get(&id) {
            match p.data {
                ScopeData::Destruction => {
                    debug!("temporary_scope({expr_id:?}) = {id:?} [enclosing]");
                    return Some(id);
                }
                _ => id = p,
            }
        }

        debug!("temporary_scope({expr_id:?}) = None");
        None
    }

    /// Make an association between a sub-expression and an extended lifetime
    pub fn record_rvalue_scope(&mut self, var: hir::ItemLocalId, lifetime: Option<Scope>) {
        debug!("record_rvalue_scope(var={var:?}, lifetime={lifetime:?})");
        if let Some(lifetime) = lifetime {
            assert!(var != lifetime.item_local_id());
        }
        self.map.insert(var, lifetime);
    }
}
