use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::is_type_lang_item;
use clippy_utils::{get_parent_expr, peel_middle_ty_refs};
use rustc_ast::util::parser::ExprPrecedence;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, LangItem, Mutability};
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_middle::ty::adjustment::{Adjust, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{GenericArg, Ty};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for redundant slicing expressions which use the full range, and
    /// do not change the type.
    ///
    /// ### Why is this bad?
    /// It unnecessarily adds complexity to the expression.
    ///
    /// ### Known problems
    /// If the type being sliced has an implementation of `Index<RangeFull>`
    /// that actually changes anything then it can't be removed. However, this would be surprising
    /// to people reading the code and should have a note with it.
    ///
    /// ### Example
    /// ```ignore
    /// fn get_slice(x: &[u32]) -> &[u32] {
    ///     &x[..]
    /// }
    /// ```
    /// Use instead:
    /// ```ignore
    /// fn get_slice(x: &[u32]) -> &[u32] {
    ///     x
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub REDUNDANT_SLICING,
    complexity,
    "redundant slicing of the whole range of a type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for slicing expressions which are equivalent to dereferencing the
    /// value.
    ///
    /// ### Why restrict this?
    /// Some people may prefer to dereference rather than slice.
    ///
    /// ### Example
    /// ```no_run
    /// let vec = vec![1, 2, 3];
    /// let slice = &vec[..];
    /// ```
    /// Use instead:
    /// ```no_run
    /// let vec = vec![1, 2, 3];
    /// let slice = &*vec;
    /// ```
    #[clippy::version = "1.61.0"]
    pub DEREF_BY_SLICING,
    restriction,
    "slicing instead of dereferencing"
}

declare_lint_pass!(RedundantSlicing => [REDUNDANT_SLICING, DEREF_BY_SLICING]);

static REDUNDANT_SLICING_LINT: (&Lint, &str) = (REDUNDANT_SLICING, "redundant slicing of the whole range");
static DEREF_BY_SLICING_LINT: (&Lint, &str) = (DEREF_BY_SLICING, "slicing when dereferencing would work");

impl<'tcx> LateLintPass<'tcx> for RedundantSlicing {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        let ctxt = expr.span.ctxt();
        if let ExprKind::AddrOf(BorrowKind::Ref, mutability, addressee) = expr.kind
            && addressee.span.ctxt() == ctxt
            && let ExprKind::Index(indexed, range, _) = addressee.kind
            && is_type_lang_item(cx, cx.typeck_results().expr_ty_adjusted(range), LangItem::RangeFull)
        {
            let (expr_ty, expr_ref_count) = peel_middle_ty_refs(cx.typeck_results().expr_ty(expr));
            let (indexed_ty, indexed_ref_count) = peel_middle_ty_refs(cx.typeck_results().expr_ty(indexed));
            let parent_expr = get_parent_expr(cx, expr);
            let needs_parens_for_prefix =
                parent_expr.is_some_and(|parent| cx.precedence(parent) > ExprPrecedence::Prefix);

            if expr_ty == indexed_ty {
                if expr_ref_count > indexed_ref_count {
                    // Indexing takes self by reference and can't return a reference to that
                    // reference as it's a local variable. The only way this could happen is if
                    // `self` contains a reference to the `Self` type. If this occurs then the
                    // lint no longer applies as it's essentially a field access, which is not
                    // redundant.
                    return;
                }
                let deref_count = indexed_ref_count - expr_ref_count;

                let ((lint, msg), reborrow_str, help_msg) = if mutability == Mutability::Mut {
                    // The slice was used to reborrow the mutable reference.
                    (DEREF_BY_SLICING_LINT, "&mut *", "reborrow the original value instead")
                } else if matches!(
                    parent_expr,
                    Some(Expr {
                        kind: ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, _),
                        ..
                    })
                ) || cx.typeck_results().expr_adjustments(expr).first().is_some_and(|a| {
                    matches!(
                        a.kind,
                        Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Mut { .. }))
                    )
                }) || (matches!(
                    cx.typeck_results().expr_ty(indexed).ref_mutability(),
                    Some(Mutability::Mut)
                ) && mutability == Mutability::Not)
                {
                    (DEREF_BY_SLICING_LINT, "&*", "reborrow the original value instead")
                } else if deref_count != 0 {
                    (DEREF_BY_SLICING_LINT, "", "dereference the original value instead")
                } else {
                    (REDUNDANT_SLICING_LINT, "", "use the original value instead")
                };

                span_lint_and_then(cx, lint, expr.span, msg, |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let snip = snippet_with_context(cx, indexed.span, ctxt, "..", &mut app).0;
                    let sugg = if (deref_count != 0 || !reborrow_str.is_empty()) && needs_parens_for_prefix {
                        format!("({reborrow_str}{}{snip})", "*".repeat(deref_count))
                    } else {
                        format!("{reborrow_str}{}{snip}", "*".repeat(deref_count))
                    };
                    diag.span_suggestion(expr.span, help_msg, sugg, app);
                });
            } else if let Some(target_id) = cx.tcx.lang_items().deref_target()
                && let Ok(deref_ty) = cx.tcx.try_normalize_erasing_regions(
                    cx.typing_env(),
                    Ty::new_projection_from_args(cx.tcx, target_id, cx.tcx.mk_args(&[GenericArg::from(indexed_ty)])),
                )
                && deref_ty == expr_ty
            {
                let (lint, msg) = DEREF_BY_SLICING_LINT;
                span_lint_and_then(cx, lint, expr.span, msg, |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let snip = snippet_with_context(cx, indexed.span, ctxt, "..", &mut app).0;
                    let sugg = if needs_parens_for_prefix {
                        format!("(&{}{}*{snip})", mutability.prefix_str(), "*".repeat(indexed_ref_count))
                    } else {
                        format!("&{}{}*{snip}", mutability.prefix_str(), "*".repeat(indexed_ref_count))
                    };
                    diag.span_suggestion(expr.span, "dereference the original value instead", sugg, app);
                });
            }
        }
    }
}
