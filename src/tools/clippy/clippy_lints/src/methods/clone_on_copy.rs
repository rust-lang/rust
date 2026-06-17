use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::peel_hir_expr_refs;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::is_copy;
use rustc_errors::Applicability;
use rustc_hir::attrs::lang_items::LangItem;
use rustc_hir::{BindingMode, ByRef, Expr, ExprKind, MatchSource, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::print::with_forced_trimmed_paths;

use super::CLONE_ON_COPY;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParentIsSuffixExpr {
    Yes,
    No,
    Return,
}

fn check_parent_is_suffix_expr(cx: &LateContext<'_>, expr: &Expr<'_>) -> ParentIsSuffixExpr {
    match cx.tcx.parent_hir_node(expr.hir_id) {
        Node::Expr(parent) => match parent.kind {
            // &*x is a nop, &x.clone() is not
            ExprKind::AddrOf(..) => ParentIsSuffixExpr::Return,
            // (*x).func() is useless, x.clone().func() can work in case func borrows self
            ExprKind::MethodCall(_, self_arg, ..)
                if expr.hir_id == self_arg.hir_id
                    && cx.typeck_results.expr_ty(expr) != cx.typeck_results.expr_ty_adjusted(expr) =>
            {
                ParentIsSuffixExpr::Return
            },
            // ? is a Call, makes sure not to rec *x?, but rather (*x)?
            ExprKind::Call(hir_callee, [_]) => {
                if matches!(
                    hir_callee.kind,
                    ExprKind::Path(qpath)
                    if cx.tcx.qpath_is_lang_item(qpath, LangItem::TryTraitBranch)
                ) {
                    ParentIsSuffixExpr::Yes
                } else {
                    ParentIsSuffixExpr::No
                }
            },
            ExprKind::MethodCall(_, self_arg, ..) if expr.hir_id == self_arg.hir_id => ParentIsSuffixExpr::Yes,
            ExprKind::Match(_, _, MatchSource::TryDesugar(_) | MatchSource::AwaitDesugar)
            | ExprKind::Field(..)
            | ExprKind::Index(..) => ParentIsSuffixExpr::Yes,
            _ => ParentIsSuffixExpr::No,
        },
        // local binding capturing a reference
        Node::LetStmt(l) if matches!(l.pat.kind, PatKind::Binding(BindingMode(ByRef::Yes(..), _), ..)) => {
            ParentIsSuffixExpr::Return
        },
        _ => ParentIsSuffixExpr::No,
    }
}

/// Checks for the `CLONE_ON_COPY` lint.
pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>) {
    if cx
        .typeck_results
        .type_dependent_def_id(expr.hir_id)
        .and_then(|id| cx.tcx.trait_of_assoc(id))
        .zip(cx.tcx.lang_items().clone_trait())
        .is_none_or(|(x, y)| x != y)
    {
        return;
    }
    let arg_adjustments = cx.typeck_results.expr_adjustments(receiver);
    let arg_ty = arg_adjustments
        .last()
        .map_or_else(|| cx.typeck_results.expr_ty(receiver), |a| a.target);

    let ty = cx.typeck_results.expr_ty(expr);
    if let ty::Ref(_, inner, _) = arg_ty.kind()
        && let ty::Ref(..) = inner.kind()
    {
        return; // don't report clone_on_copy
    }

    if !is_copy(cx, ty) {
        return;
    }
    let check_res = check_parent_is_suffix_expr(cx, expr);
    if check_res == ParentIsSuffixExpr::Return {
        return;
    }

    let mut app = Applicability::MachineApplicable;
    let snip = snippet_with_context(cx, receiver.span, expr.span.ctxt(), "_", &mut app).0;

    let deref_count = arg_adjustments
        .iter()
        .take_while(|adj| matches!(adj.kind, Adjust::Deref(_)))
        .count();
    let (help, sugg) = if deref_count == 0 {
        ("try removing the `clone` call", snip.into())
    } else if check_res == ParentIsSuffixExpr::Yes {
        ("try dereferencing it", format!("({}{snip})", "*".repeat(deref_count)))
    } else {
        ("try dereferencing it", format!("{}{snip}", "*".repeat(deref_count)))
    };

    span_lint_and_sugg(
        cx,
        CLONE_ON_COPY,
        expr.span,
        with_forced_trimmed_paths!(format!(
            "using `clone` on type `{ty}` which implements the `Copy` trait"
        )),
        help,
        sugg,
        app,
    );
}

pub(super) fn check_function(cx: &LateContext<'_>, expr: &Expr<'_>) {
    let ExprKind::Call(func, args) = expr.kind else { return };

    if let ExprKind::Path(qpath) = func.kind
        && let Some(def_id) = cx.typeck_results.qpath_res(&qpath, func.hir_id).opt_def_id()
        && cx.tcx.trait_of_assoc(def_id) == cx.tcx.lang_items().clone_trait()
        && let [arg] = args
        && let ty = cx.typeck_results.expr_ty(expr)
    {
        if !is_copy(cx, ty) {
            return;
        }

        let check_res = check_parent_is_suffix_expr(cx, expr);
        if check_res == ParentIsSuffixExpr::Return {
            return;
        }

        // Peel the & operator in the argument.
        let (peeled_arg, ref_count) = peel_hir_expr_refs(arg);

        let mut app = Applicability::MachineApplicable;
        let snip = snippet_with_context(cx, peeled_arg.span, func.span.ctxt(), "_", &mut app).0;

        let arg_adjustments = cx.typeck_results.expr_adjustments(arg);
        let deref_count = arg_adjustments
            .iter()
            .take_while(|adj| matches!(adj.kind, Adjust::Deref(_)))
            .count()
            - ref_count;

        let (help, sugg) = if deref_count == 0 {
            if check_res == ParentIsSuffixExpr::Yes && snip.starts_with('*') {
                ("try dereferencing it", format!("({snip})"))
            } else {
                ("try removing the `clone` call", snip.into())
            }
        } else if check_res == ParentIsSuffixExpr::Yes {
            // deref_count comes from the original argument `&x`, so we need to decrement `deref_count`.
            ("try dereferencing it", format!("({}{snip})", "*".repeat(deref_count)))
        } else {
            ("try dereferencing it", format!("{}{snip}", "*".repeat(deref_count)))
        };

        span_lint_and_sugg(
            cx,
            CLONE_ON_COPY,
            expr.span,
            with_forced_trimmed_paths!(format!(
                "using `clone` on type `{ty}` which implements the `Copy` trait"
            )),
            help,
            sugg,
            app,
        );
    }
}
