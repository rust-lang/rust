use rustc_hir as hir;
use rustc_lint_defs::builtin::STATIC_MUT_REFS;
use rustc_middle::ty::{Mutability, TyCtxt};
use rustc_span::Span;

use crate::errors;

/// Check for shared or mutable references of `static mut` inside expression
pub(crate) fn maybe_expr_static_mut(tcx: TyCtxt<'_>, expr: hir::Expr<'_>) {
    let span = expr.span;
    let hir_id = expr.hir_id;
    if let hir::ExprKind::AddrOf(borrow_kind, m, expr) = expr.kind
        && matches!(borrow_kind, hir::BorrowKind::Ref)
        && path_if_static_mut(expr)
    {
        handle_static_mut_ref(
            tcx,
            span,
            span.with_hi(expr.span.lo()),
            span.shrink_to_hi(),
            span.edition().at_least_rust_2024(),
            m,
            hir_id,
        );
    }
}

/// Check for shared or mutable references of `static mut` inside statement
pub(crate) fn maybe_stmt_static_mut(tcx: TyCtxt<'_>, stmt: hir::Stmt<'_>) {
    if let hir::StmtKind::Let(loc) = stmt.kind
        && let hir::PatKind::Binding(ba, _, _, _) = loc.pat.kind
        && let hir::ByRef::Yes(rmutbl) = ba.0
        && let Some(init) = loc.init
        && path_if_static_mut(init)
    {
        handle_static_mut_ref(
            tcx,
            init.span,
            init.span.shrink_to_lo(),
            init.span.shrink_to_hi(),
            loc.span.edition().at_least_rust_2024(),
            rmutbl,
            stmt.hir_id,
        );
    }
}

fn path_if_static_mut(expr: &hir::Expr<'_>) -> bool {
    if let hir::ExprKind::Path(qpath) = expr.kind
        && let hir::QPath::Resolved(_, path) = qpath
        && let hir::def::Res::Def(def_kind, _) = path.res
        && let hir::def::DefKind::Static { safety: _, mutability: Mutability::Mut, nested: false } =
            def_kind
    {
        return true;
    }
    false
}

fn handle_static_mut_ref(
    tcx: TyCtxt<'_>,
    span: Span,
    lo: Span,
    hi: Span,
    e2024: bool,
    mutable: Mutability,
    hir_id: hir::HirId,
) {
    if e2024 {
        let (sugg, shared) = if mutable == Mutability::Mut {
            (errors::MutRefSugg::Mut { lo, hi }, "mutable")
        } else {
            (errors::MutRefSugg::Shared { lo, hi }, "shared")
        };
        tcx.dcx().emit_err(errors::StaticMutRef { span, sugg, shared });
    } else {
        let (sugg, shared) = if mutable == Mutability::Mut {
            (errors::MutRefSugg::Mut { lo, hi }, "mutable")
        } else {
            (errors::MutRefSugg::Shared { lo, hi }, "shared")
        };
        tcx.emit_node_span_lint(STATIC_MUT_REFS, hir_id, span, errors::RefOfMutStatic {
            span,
            sugg,
            shared,
        });
    }
}
