use rustc_hir as hir;
use rustc_hir::ItemLocalMap;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use tracing::debug;

use crate::middle::region::{Scope, ScopeData, ScopeTree};

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
    /// It also emits a lint on potential backwards incompatible change to the temporary scope
    /// which is *for now* always shortening.
    pub fn temporary_scope(
        &self,
        region_scope_tree: &ScopeTree,
        expr_id: hir::ItemLocalId,
    ) -> (Option<Scope>, Option<Scope>) {
        // Check for a designated rvalue scope.
        if let Some(&s) = self.map.get(&expr_id) {
            debug!("temporary_scope({expr_id:?}) = {s:?} [custom]");
            return (s, None);
        }

        // Otherwise, locate the innermost terminating scope
        // if there's one. Static items, for instance, won't
        // have an enclosing scope, hence no scope will be
        // returned.
        let mut id = Scope { local_id: expr_id, data: ScopeData::Node };
        let mut backwards_incompatible = None;

        while let Some(&p) = region_scope_tree.parent_map.get(&id) {
            match p.data {
                ScopeData::Destruction => {
                    debug!("temporary_scope({expr_id:?}) = {id:?} [enclosing]");
                    return (Some(id), backwards_incompatible);
                }
                ScopeData::IfThenRescope => {
                    debug!("temporary_scope({expr_id:?}) = {p:?} [enclosing]");
                    return (Some(p), backwards_incompatible);
                }
                ScopeData::Node
                | ScopeData::CallSite
                | ScopeData::Arguments
                | ScopeData::IfThen
                | ScopeData::Remainder(_) => {
                    // If we haven't already passed through a backwards-incompatible node,
                    // then check if we are passing through one now and record it if so.
                    // This is for now only working for cases where a temporary lifetime is
                    // *shortened*.
                    if backwards_incompatible.is_none() {
                        backwards_incompatible = region_scope_tree
                            .backwards_incompatible_scope
                            .get(&p.local_id)
                            .copied();
                    }
                    id = p
                }
            }
        }

        debug!("temporary_scope({expr_id:?}) = None");
        (None, backwards_incompatible)
    }

    /// Make an association between a sub-expression and an extended lifetime
    pub fn record_rvalue_scope(&mut self, var: hir::ItemLocalId, lifetime: Option<Scope>) {
        debug!("record_rvalue_scope(var={var:?}, lifetime={lifetime:?})");
        if let Some(lifetime) = lifetime {
            assert!(var != lifetime.local_id);
        }
        self.map.insert(var, lifetime);
    }
}
