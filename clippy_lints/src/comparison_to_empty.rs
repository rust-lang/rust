use crate::utils::{snippet_with_applicability, span_lint_and_sugg};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::{BinOpKind, Expr, ExprKind, ItemKind, TraitItemRef};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::{Span, Spanned};

declare_clippy_lint! {
    /// **What it does:**
    ///
    /// **Why is this bad?**
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // example code where clippy issues a warning
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// ```
    pub COMPARISON_TO_EMPTY,
    style,
    "default lint description"
}

declare_lint_pass!(ComparisonToEmpty => [COMPARISON_TO_EMPTY]);

impl LateLintPass<'_> for ComparisonToEmpty {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node: cmp, .. }, ref left, ref right) = expr.kind {
            match cmp {
                BinOpKind::Eq => {
                    check_cmp(cx, expr.span, left, right, "", 0); // len == 0
                    check_cmp(cx, expr.span, right, left, "", 0); // 0 == len
                },
                BinOpKind::Ne => {
                    check_cmp(cx, expr.span, left, right, "!", 0); // len != 0
                    check_cmp(cx, expr.span, right, left, "!", 0); // 0 != len
                },
                BinOpKind::Gt => {
                    check_cmp(cx, expr.span, left, right, "!", 0); // len > 0
                    check_cmp(cx, expr.span, right, left, "", 1); // 1 > len
                },
                BinOpKind::Lt => {
                    check_cmp(cx, expr.span, left, right, "", 1); // len < 1
                    check_cmp(cx, expr.span, right, left, "!", 0); // 0 < len
                },
                BinOpKind::Ge => check_cmp(cx, expr.span, left, right, "!", 1), // len >= 1
                BinOpKind::Le => check_cmp(cx, expr.span, right, left, "!", 1), // 1 <= len
                _ => (),
            }
        }
    }

}


fn check_cmp(cx: &LateContext<'_>, span: Span, lit1: &Expr<'_>, lit2: &Expr<'_>, op: &str, compare_to: u32) {
    check_empty_expr(cx, span, lit1, lit2, op)
}

fn check_empty_expr(
    cx: &LateContext<'_>,
    span: Span,
    lit1: &Expr<'_>,
    lit2: &Expr<'_>,
    op: &str
) {
    if (is_empty_array(lit2) || is_empty_string(lit2)) && has_is_empty(cx, lit1) {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            COMPARISON_TO_EMPTY,
            span,
            &format!("comparison to empty slice"),
            &format!("using `{}is_empty` is clearer and more explicit", op),
            format!(
                "{}{}.is_empty()",
                op,
                snippet_with_applicability(cx, lit1.span, "_", &mut applicability)
            ),
            applicability,
        );
    }
}

fn is_empty_string(expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(ref lit) = expr.kind {
        if let LitKind::Str(lit, _) = lit.node {
            let lit = lit.as_str();
            return lit == "";
        }
    }
    false
}

fn is_empty_array(expr: &Expr<'_>) -> bool {
    if let ExprKind::Array(ref arr) = expr.kind {
        return arr.is_empty();
    }
    false
}


/// Checks if this type has an `is_empty` method.
fn has_is_empty(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    /// Gets an `AssocItem` and return true if it matches `is_empty(self)`.
    fn is_is_empty(cx: &LateContext<'_>, item: &ty::AssocItem) -> bool {
        if let ty::AssocKind::Fn = item.kind {
            if item.ident.name.as_str() == "is_empty" {
                let sig = cx.tcx.fn_sig(item.def_id);
                let ty = sig.skip_binder();
                ty.inputs().len() == 1
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Checks the inherent impl's items for an `is_empty(self)` method.
    fn has_is_empty_impl(cx: &LateContext<'_>, id: DefId) -> bool {
        cx.tcx.inherent_impls(id).iter().any(|imp| {
            cx.tcx
                .associated_items(*imp)
                .in_definition_order()
                .any(|item| is_is_empty(cx, &item))
        })
    }

    let ty = &cx.typeck_results().expr_ty(expr).peel_refs();
    match ty.kind() {
        ty::Dynamic(ref tt, ..) => tt.principal().map_or(false, |principal| {
            cx.tcx
                .associated_items(principal.def_id())
                .in_definition_order()
                .any(|item| is_is_empty(cx, &item))
        }),
        ty::Projection(ref proj) => has_is_empty_impl(cx, proj.item_def_id),
        ty::Adt(id, _) => has_is_empty_impl(cx, id.did),
        ty::Array(..) | ty::Slice(..) | ty::Str => true,
        _ => false,
    }
}
