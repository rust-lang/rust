use rustc_hir as hir;
use rustc_hir::ItemLocalMap;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use tracing::debug;

use crate::middle::region::{ScopeCompatibility, Scope, ScopeData, ScopeTree};
use crate::mir::BackwardIncompatibleDropReason;

/// `RvalueScopes` is a mapping from sub-expressions to _extended_ lifetime as determined by
/// rules laid out in `rustc_hir_analysis::check::rvalue_scopes`.
#[derive(TyEncodable, TyDecodable, Clone, Debug, Default, Eq, PartialEq, HashStable)]
pub struct RvalueScopes {
    map: ItemLocalMap<(Option<Scope>, Option<(Scope, BackwardIncompatibleDropReason)>)>,
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
    ) -> (Option<Scope>, Option<(Scope, BackwardIncompatibleDropReason)>) {
        // Check for a designated rvalue scope.
        if let Some(&(s, future_scope)) = self.map.get(&expr_id) {
            debug!("temporary_scope({expr_id:?}) = {s:?} [custom]");
            return (s, future_scope);
        }

        // Otherwise, locate the innermost terminating scope
        // if there's one. Static items, for instance, won't
        // have an enclosing scope, hence no scope will be
        // returned.
        region_scope_tree
            .default_temporary_scope(Scope { local_id: expr_id, data: ScopeData::Node })
    }

    /// Make an association between a sub-expression and an extended lifetime
    pub fn record_rvalue_scope(
        &mut self,
        var: hir::ItemLocalId,
        lifetime: Option<Scope>,
        compat: ScopeCompatibility,
    ) {
        debug!("record_rvalue_scope(var={var:?}, lifetime={lifetime:?})");
        if let Some(lifetime) = lifetime {
            assert!(var != lifetime.local_id);
        }
        let future_scope =
            if let ScopeCompatibility::FutureIncompatible { shortens_to } = compat
            {
                Some((shortens_to, BackwardIncompatibleDropReason::MacroExtendedScope))
            } else {
                None
            };
        self.map.insert(var, (lifetime, future_scope));
    }
}
