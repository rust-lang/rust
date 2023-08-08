use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_node;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::is_copy;
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, ByRef, Expr, ExprKind, MatchSource, Node, PatKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::{self};
use rustc_span::symbol::{sym, Symbol};

use super::CLONE_ON_COPY;

/// Checks for the `CLONE_ON_COPY` lint.
#[allow(clippy::too_many_lines)]
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    method_name: Symbol,
    receiver: &Expr<'_>,
    args: &[Expr<'_>],
) {
    let arg = if method_name == sym::clone && args.is_empty() {
        receiver
    } else {
        return;
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
        if let ty::Ref(..) = inner.kind() {
            return; // don't report clone_on_copy
        }
    }

    if is_copy(cx, ty) {
        let parent_is_suffix_expr = match get_parent_node(cx.tcx, expr.hir_id) {
            Some(Node::Expr(parent)) => match parent.kind {
                // &*x is a nop, &x.clone() is not
                ExprKind::AddrOf(..) => return,
                // (*x).func() is useless, x.clone().func() can work in case func borrows self
                ExprKind::MethodCall(_, self_arg, ..)
                    if expr.hir_id == self_arg.hir_id && ty != cx.typeck_results().expr_ty_adjusted(expr) =>
                {
                    return;
                },
                // ? is a Call, makes sure not to rec *x?, but rather (*x)?
                ExprKind::Call(hir_callee, _) => matches!(
                    hir_callee.kind,
                    ExprKind::Path(QPath::LangItem(rustc_hir::LangItem::TryTraitBranch, _, _))
                ),
                ExprKind::MethodCall(_, self_arg, ..) if expr.hir_id == self_arg.hir_id => true,
                ExprKind::Match(_, _, MatchSource::TryDesugar | MatchSource::AwaitDesugar)
                | ExprKind::Field(..)
                | ExprKind::Index(..) => true,
                _ => false,
            },
            // local binding capturing a reference
            Some(Node::Local(l)) if matches!(l.pat.kind, PatKind::Binding(BindingAnnotation(ByRef::Yes, _), ..)) => {
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
            ("try dereferencing it", format!("({}{snip})", "*".repeat(deref_count)))
        } else {
            ("try dereferencing it", format!("{}{snip}", "*".repeat(deref_count)))
        };

        span_lint_and_sugg(
            cx,
            CLONE_ON_COPY,
            expr.span,
            &with_forced_trimmed_paths!(format!(
                "using `clone` on type `{ty}` which implements the `Copy` trait"
            )),
            help,
            sugg,
            app,
        );
    }
}
