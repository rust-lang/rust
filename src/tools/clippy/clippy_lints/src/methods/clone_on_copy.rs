use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::get_parent_node;
use clippy_utils::source::snippet_with_context;
use clippy_utils::sugg;
use clippy_utils::ty::is_copy;
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, Expr, ExprKind, MatchSource, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, adjustment::Adjust};
use rustc_span::symbol::{sym, Symbol};

use super::CLONE_DOUBLE_REF;
use super::CLONE_ON_COPY;

/// Checks for the `CLONE_ON_COPY` lint.
#[allow(clippy::too_many_lines)]
pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, method_name: Symbol, args: &[Expr<'_>]) {
    let arg = match args {
        [arg] if method_name == sym::clone => arg,
        _ => return,
    };
    if cx
        .typeck_results()
        .type_dependent_def_id(expr.hir_id)
        .and_then(|id| cx.tcx.trait_of_item(id))
        .zip(cx.tcx.lang_items().clone_trait())
        .map_or(true, |(x, y)| x != y)
    {
        return;
    }
    let arg_adjustments = cx.typeck_results().expr_adjustments(arg);
    let arg_ty = arg_adjustments
        .last()
        .map_or_else(|| cx.typeck_results().expr_ty(arg), |a| a.target);

    let ty = cx.typeck_results().expr_ty(expr);
    if let ty::Ref(_, inner, _) = arg_ty.kind() {
        if let ty::Ref(_, innermost, _) = inner.kind() {
            span_lint_and_then(
                cx,
                CLONE_DOUBLE_REF,
                expr.span,
                &format!(
                    "using `clone` on a double-reference; \
                    this will copy the reference of type `{}` instead of cloning the inner type",
                    ty
                ),
                |diag| {
                    if let Some(snip) = sugg::Sugg::hir_opt(cx, arg) {
                        let mut ty = innermost;
                        let mut n = 0;
                        while let ty::Ref(_, inner, _) = ty.kind() {
                            ty = inner;
                            n += 1;
                        }
                        let refs = "&".repeat(n + 1);
                        let derefs = "*".repeat(n);
                        let explicit = format!("<{}{}>::clone({})", refs, ty, snip);
                        diag.span_suggestion(
                            expr.span,
                            "try dereferencing it",
                            format!("{}({}{}).clone()", refs, derefs, snip.deref()),
                            Applicability::MaybeIncorrect,
                        );
                        diag.span_suggestion(
                            expr.span,
                            "or try being explicit if you are sure, that you want to clone a reference",
                            explicit,
                            Applicability::MaybeIncorrect,
                        );
                    }
                },
            );
            return; // don't report clone_on_copy
        }
    }

    if is_copy(cx, ty) {
        let parent_is_suffix_expr = match get_parent_node(cx.tcx, expr.hir_id) {
            Some(Node::Expr(parent)) => match parent.kind {
                // &*x is a nop, &x.clone() is not
                ExprKind::AddrOf(..) => return,
                // (*x).func() is useless, x.clone().func() can work in case func borrows self
                ExprKind::MethodCall(_, [self_arg, ..], _)
                    if expr.hir_id == self_arg.hir_id && ty != cx.typeck_results().expr_ty_adjusted(expr) =>
                {
                    return;
                },
                ExprKind::MethodCall(_, [self_arg, ..], _) if expr.hir_id == self_arg.hir_id => true,
                ExprKind::Match(_, _, MatchSource::TryDesugar | MatchSource::AwaitDesugar)
                | ExprKind::Field(..)
                | ExprKind::Index(..) => true,
                _ => false,
            },
            // local binding capturing a reference
            Some(Node::Local(l))
                if matches!(
                    l.pat.kind,
                    PatKind::Binding(BindingAnnotation::Ref | BindingAnnotation::RefMut, ..)
                ) =>
            {
                return;
            },
            _ => false,
        };

        let mut app = Applicability::MachineApplicable;
        let snip = snippet_with_context(cx, arg.span, expr.span.ctxt(), "_", &mut app).0;

        let deref_count = arg_adjustments
            .iter()
            .take_while(|adj| matches!(adj.kind, Adjust::Deref(_)))
            .count();
        let (help, sugg) = if deref_count == 0 {
            ("try removing the `clone` call", snip.into())
        } else if parent_is_suffix_expr {
            ("try dereferencing it", format!("({}{})", "*".repeat(deref_count), snip))
        } else {
            ("try dereferencing it", format!("{}{}", "*".repeat(deref_count), snip))
        };

        span_lint_and_sugg(
            cx,
            CLONE_ON_COPY,
            expr.span,
            &format!("using `clone` on type `{}` which implements the `Copy` trait", ty),
            help,
            sugg,
            app,
        );
    }
}
