use rustc_hir as hir;
use rustc_hir_pretty::qpath_to_string;
use rustc_lint_defs::builtin::STATIC_MUT_REFS;
use rustc_middle::ty::{Mutability, TyCtxt};
use rustc_span::symbol::Ident;
use rustc_span::Span;

use crate::errors;

/// Check for shared or mutable references of `static mut` inside expression
pub fn maybe_expr_static_mut(tcx: TyCtxt<'_>, expr: &hir::Expr<'_>) {
    let err_span = expr.span;
    let lint_level_hir_id = expr.hir_id;
    match expr.kind {
        hir::ExprKind::AddrOf(borrow_kind, m, ex)
            if matches!(borrow_kind, hir::BorrowKind::Ref)
                && let Some((var, err_span)) = path_if_static_mut(tcx, ex, err_span) =>
        {
            handle_static_mut_ref(
                tcx,
                err_span,
                var,
                err_span.edition().at_least_rust_2024(),
                m,
                lint_level_hir_id,
                !expr.span.from_expansion(),
            );
        }
        hir::ExprKind::Index(expr, _, _)
            if let Some((var, err_span)) = path_if_static_mut(tcx, expr, err_span) =>
        {
            handle_static_mut_ref(
                tcx,
                err_span,
                var,
                err_span.edition().at_least_rust_2024(),
                Mutability::Not,
                lint_level_hir_id,
                false,
            );
        }
        _ => {}
    }
}

/// Check for shared or mutable references of `static mut` inside statement
pub fn maybe_stmt_static_mut(tcx: TyCtxt<'_>, stmt: &hir::Stmt<'_>) {
    if let hir::StmtKind::Let(loc) = stmt.kind
        && let hir::PatKind::Binding(ba, _, _, _) = loc.pat.kind
        && let hir::ByRef::Yes(rmutbl) = ba.0
        && let Some(init) = loc.init
        && let Some((var, err_span)) = path_if_static_mut(tcx, init, init.span)
    {
        handle_static_mut_ref(
            tcx,
            err_span,
            var,
            loc.span.edition().at_least_rust_2024(),
            rmutbl,
            stmt.hir_id,
            false,
        );
    }
}

/// Check if method call takes shared or mutable references of `static mut`
pub fn maybe_method_static_mut(tcx: TyCtxt<'_>, expr: &hir::Expr<'_>) {
    if let hir::ExprKind::MethodCall(_, e, _, _) = expr.kind
        && let Some((var, err_span)) = path_if_static_mut(tcx, e, expr.span)
        && let typeck = tcx.typeck(expr.hir_id.owner)
        && let Some(method_def_id) = typeck.type_dependent_def_id(expr.hir_id)
        && let inputs = tcx.fn_sig(method_def_id).skip_binder().inputs().skip_binder()
        && !inputs.is_empty()
        && inputs[0].is_ref()
    {
        handle_static_mut_ref(
            tcx,
            err_span,
            var,
            err_span.edition().at_least_rust_2024(),
            Mutability::Not,
            expr.hir_id,
            false,
        );
    }
}

fn path_if_static_mut(
    tcx: TyCtxt<'_>,
    mut expr: &hir::Expr<'_>,
    mut err_span: Span,
) -> Option<(String, Span)> {
    if err_span.from_expansion() {
        err_span = expr.span;
    }

    let mut fields: Vec<Ident> = Vec::new();
    while let hir::ExprKind::Field(e, ident) = expr.kind {
        expr = e;
        fields.push(ident);
    }

    if let hir::ExprKind::Path(qpath) = expr.kind
        && let hir::QPath::Resolved(_, path) = qpath
        && let hir::def::Res::Def(def_kind, _) = path.res
        && let hir::def::DefKind::Static { safety: _, mutability: Mutability::Mut, nested: false } =
            def_kind
    {
        let mut var = qpath_to_string(&tcx, &qpath);
        for field_ident in fields.iter().rev() {
            var.push_str(".");
            var.push_str(field_ident.as_str());
        }
        return Some((var, err_span));
    }
    None
}

fn handle_static_mut_ref(
    tcx: TyCtxt<'_>,
    err_span: Span,
    var: String,
    e2024: bool,
    mutable: Mutability,
    lint_level_hir_id: hir::HirId,
    suggest_addr_of: bool,
) {
    let shared = if mutable == Mutability::Mut { "mutable" } else { "shared" };
    let sugg = suggest_addr_of.then(|| {
        if matches!(mutable, Mutability::Mut) {
            errors::StaticMutRefSugg::Mut { span: err_span, var }
        } else {
            errors::StaticMutRefSugg::Shared { span: err_span, var }
        }
    });
    if e2024 {
        tcx.dcx().emit_err(errors::StaticMutRef { span: err_span, sugg, shared });
        return;
    }
    tcx.emit_node_span_lint(
        STATIC_MUT_REFS,
        lint_level_hir_id,
        err_span,
        errors::RefOfMutStatic { span: err_span, sugg, shared },
    );
}
