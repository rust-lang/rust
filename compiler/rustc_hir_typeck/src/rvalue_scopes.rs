use hir::Node;
use hir::def_id::DefId;
use rustc_hir as hir;
use rustc_middle::bug;
use rustc_middle::middle::region::{RvalueCandidate, Scope, ScopeTree};
use rustc_middle::ty::RvalueScopes;
use tracing::debug;

use super::FnCtxt;

/// Applied to an expression `expr` if `expr` -- or something owned or partially owned by
/// `expr` -- is going to be indirectly referenced by a variable in a let statement. In that
/// case, the "temporary lifetime" or `expr` is extended to be the block enclosing the `let`
/// statement.
///
/// More formally, if `expr` matches the grammar `ET`, record the rvalue scope of the matching
/// `<rvalue>` as `blk_id`:
///
/// ```text
///     ET = *ET
///        | ET[...]
///        | ET.f
///        | (ET)
///        | <rvalue>
/// ```
///
/// Note: ET is intended to match "rvalues or places based on rvalues".
fn record_rvalue_scope_rec(
    rvalue_scopes: &mut RvalueScopes,
    mut expr: &hir::Expr<'_>,
    lifetime: Option<Scope>,
) {
    loop {
        // Note: give all the expressions matching `ET` with the
        // extended temporary lifetime, not just the innermost rvalue,
        // because in codegen if we must compile e.g., `*rvalue()`
        // into a temporary, we request the temporary scope of the
        // outer expression.

        rvalue_scopes.record_rvalue_scope(expr.hir_id.local_id, lifetime);

        match expr.kind {
            hir::ExprKind::AddrOf(_, _, subexpr)
            | hir::ExprKind::Unary(hir::UnOp::Deref, subexpr)
            | hir::ExprKind::Field(subexpr, _)
            | hir::ExprKind::Index(subexpr, _, _) => {
                expr = subexpr;
            }
            _ => {
                return;
            }
        }
    }
}
fn record_rvalue_scope(
    rvalue_scopes: &mut RvalueScopes,
    expr: &hir::Expr<'_>,
    candidate: &RvalueCandidate,
) {
    debug!("resolve_rvalue_scope(expr={expr:?}, candidate={candidate:?})");
    record_rvalue_scope_rec(rvalue_scopes, expr, candidate.lifetime)
    // FIXME(@dingxiangfei2009): handle the candidates in the function call arguments
}

pub(crate) fn resolve_rvalue_scopes<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    scope_tree: &'a ScopeTree,
    def_id: DefId,
) -> RvalueScopes {
    let tcx = &fcx.tcx;
    let mut rvalue_scopes = RvalueScopes::new();
    debug!("start resolving rvalue scopes, def_id={def_id:?}");
    debug!("rvalue_scope: rvalue_candidates={:?}", scope_tree.rvalue_candidates);
    for (&hir_id, candidate) in &scope_tree.rvalue_candidates {
        let Node::Expr(expr) = tcx.hir_node(hir_id) else { bug!("hir node does not exist") };
        record_rvalue_scope(&mut rvalue_scopes, expr, candidate);
    }
    rvalue_scopes
}
