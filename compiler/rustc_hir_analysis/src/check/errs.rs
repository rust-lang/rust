use rustc_hir as hir;
use rustc_lint_defs::builtin::STATIC_MUT_REFS;
use rustc_middle::ty::{Mutability, TyCtxt, TyKind};
use rustc_span::Span;

use crate::errors;

/// Check for shared or mutable references of `static mut` inside expression
pub fn maybe_expr_static_mut(tcx: TyCtxt<'_>, expr: &hir::Expr<'_>) {
    let err_span = expr.span;
    let lint_level_hir_id = expr.hir_id;
    match expr.kind {
        hir::ExprKind::AddrOf(borrow_kind, m, ex)
            if matches!(borrow_kind, hir::BorrowKind::Ref)
                && let Some(err_span) = path_is_static_mut(ex, err_span) =>
        {
            handle_static_mut_ref(
                tcx,
                err_span,
                err_span.with_hi(ex.span.lo()),
                err_span.shrink_to_hi(),
                err_span.edition().at_least_rust_2024(),
                Some(m),
                lint_level_hir_id,
                !expr.span.from_expansion(),
            );
        }
        hir::ExprKind::Index(expr, _, _)
            if let Some(err_span) = path_is_static_mut(expr, err_span) =>
        {
            handle_static_mut_ref(
                tcx,
                err_span,
                err_span.with_hi(expr.span.lo()),
                err_span.shrink_to_hi(),
                err_span.edition().at_least_rust_2024(),
                None,
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
        && let Some(err_span) = path_is_static_mut(init, init.span)
    {
        handle_static_mut_ref(
            tcx,
            err_span,
            err_span.shrink_to_lo(),
            err_span.shrink_to_hi(),
            loc.span.edition().at_least_rust_2024(),
            Some(rmutbl),
            stmt.hir_id,
            false,
        );
    }
}

/// Check if method call takes shared or mutable references of `static mut`
#[allow(rustc::usage_of_ty_tykind)]
pub fn maybe_method_static_mut(tcx: TyCtxt<'_>, expr: &hir::Expr<'_>) {
    if let hir::ExprKind::MethodCall(_, e, _, _) = expr.kind
        && let Some(err_span) = path_is_static_mut(e, expr.span)
        && let typeck = tcx.typeck(expr.hir_id.owner)
        && let Some(method_def_id) = typeck.type_dependent_def_id(expr.hir_id)
        && let inputs = tcx.fn_sig(method_def_id).skip_binder().inputs().skip_binder()
        && !inputs.is_empty()
        && inputs[0].is_ref()
    {
        let m = if let TyKind::Ref(_, _, m) = inputs[0].kind() { *m } else { Mutability::Not };

        handle_static_mut_ref(
            tcx,
            err_span,
            err_span.shrink_to_lo(),
            err_span.shrink_to_hi(),
            err_span.edition().at_least_rust_2024(),
            Some(m),
            expr.hir_id,
            false,
        );
    }
}

fn path_is_static_mut(mut expr: &hir::Expr<'_>, mut err_span: Span) -> Option<Span> {
    if err_span.from_expansion() {
        err_span = expr.span;
    }

    while let hir::ExprKind::Field(e, _) = expr.kind {
        expr = e;
    }

    if let hir::ExprKind::Path(qpath) = expr.kind
        && let hir::QPath::Resolved(_, path) = qpath
        && let hir::def::Res::Def(def_kind, _) = path.res
        && let hir::def::DefKind::Static { safety: _, mutability: Mutability::Mut, nested: false } =
            def_kind
    {
        return Some(err_span);
    }
    None
}

fn handle_static_mut_ref(
    tcx: TyCtxt<'_>,
    span: Span,
    lo: Span,
    hi: Span,
    e2024: bool,
    mutable: Option<Mutability>,
    lint_level_hir_id: hir::HirId,
    suggest_addr_of: bool,
) {
    let (shared_label, shared_note, mut_note, sugg) = match mutable {
        Some(Mutability::Mut) => {
            let sugg =
                if suggest_addr_of { Some(errors::MutRefSugg::Mut { lo, hi }) } else { None };
            ("mutable ", false, true, sugg)
        }
        Some(Mutability::Not) => {
            let sugg =
                if suggest_addr_of { Some(errors::MutRefSugg::Shared { lo, hi }) } else { None };
            ("shared ", true, false, sugg)
        }
        None => ("", true, true, None),
    };
    if e2024 {
        tcx.dcx().emit_err(errors::StaticMutRef {
            span,
            sugg,
            shared_label,
            shared_note,
            mut_note,
        });
    } else {
        tcx.emit_node_span_lint(
            STATIC_MUT_REFS,
            lint_level_hir_id,
            span,
            errors::RefOfMutStatic { span, sugg, shared_label, shared_note, mut_note },
        );
    }
}
