use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{is_type_lang_item, peel_mid_ty_refs};
use if_chain::if_chain;
use rustc_ast::util::parser::PREC_PREFIX;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, LangItem, Mutability};
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_middle::ty::Ty;
use rustc_middle::ty::adjustment::{Adjust, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::subst::GenericArg;
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
    /// ### Why is this bad?
    /// Some people may prefer to dereference rather than slice.
    ///
    /// ### Example
    /// ```rust
    /// let vec = vec![1, 2, 3];
    /// let slice = &vec[..];
    /// ```
    /// Use instead:
    /// ```rust
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
        if_chain! {
            if let ExprKind::AddrOf(BorrowKind::Ref, mutability, addressee) = expr.kind;
            if addressee.span.ctxt() == ctxt;
            if let ExprKind::Index(indexed, range) = addressee.kind;
            if is_type_lang_item(cx, cx.typeck_results().expr_ty_adjusted(range), LangItem::RangeFull);
            then {
                let (expr_ty, expr_ref_count) = peel_mid_ty_refs(cx.typeck_results().expr_ty(expr));
                let (indexed_ty, indexed_ref_count) = peel_mid_ty_refs(cx.typeck_results().expr_ty(indexed));
                let parent_expr = get_parent_expr(cx, expr);
                let needs_parens_for_prefix = parent_expr.map_or(false, |parent| {
                    parent.precedence().order() > PREC_PREFIX
                });
                let mut app = Applicability::MachineApplicable;

                let ((lint, msg), help, sugg) = if expr_ty == indexed_ty {
                    if expr_ref_count > indexed_ref_count {
                        // Indexing takes self by reference and can't return a reference to that
                        // reference as it's a local variable. The only way this could happen is if
                        // `self` contains a reference to the `Self` type. If this occurs then the
                        // lint no longer applies as it's essentially a field access, which is not
                        // redundant.
                        return;
                    }
                    let deref_count = indexed_ref_count - expr_ref_count;

                    let (lint, reborrow_str, help_str) = if mutability == Mutability::Mut {
                        // The slice was used to reborrow the mutable reference.
                        (DEREF_BY_SLICING_LINT, "&mut *", "reborrow the original value instead")
                    } else if matches!(
                        parent_expr,
                        Some(Expr {
                            kind: ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, _),
                            ..
                        })
                    ) || cx.typeck_results().expr_adjustments(expr).first().map_or(false, |a| {
                        matches!(a.kind, Adjust::Borrow(AutoBorrow::Ref(_, AutoBorrowMutability::Mut { .. })))
                    }) {
                        // The slice was used to make a temporary reference.
                        (DEREF_BY_SLICING_LINT, "&*", "reborrow the original value instead")
                    } else if deref_count != 0 {
                        (DEREF_BY_SLICING_LINT, "", "dereference the original value instead")
                    } else {
                        (REDUNDANT_SLICING_LINT, "", "use the original value instead")
                    };

                    let snip = snippet_with_context(cx, indexed.span, ctxt, "..", &mut app).0;
                    let sugg = if (deref_count != 0 || !reborrow_str.is_empty()) && needs_parens_for_prefix {
                        format!("({reborrow_str}{}{snip})", "*".repeat(deref_count))
                    } else {
                        format!("{reborrow_str}{}{snip}", "*".repeat(deref_count))
                    };

                    (lint, help_str, sugg)
                } else if let Some(target_id) = cx.tcx.lang_items().deref_target() {
                    if let Ok(deref_ty) = cx.tcx.try_normalize_erasing_regions(
                        cx.param_env,
                        Ty::new_projection(cx.tcx,target_id, cx.tcx.mk_substs(&[GenericArg::from(indexed_ty)])),
                    ) {
                        if deref_ty == expr_ty {
                            let snip = snippet_with_context(cx, indexed.span, ctxt, "..", &mut app).0;
                            let sugg = if needs_parens_for_prefix {
                                format!("(&{}{}*{snip})", mutability.prefix_str(), "*".repeat(indexed_ref_count))
                            } else {
                                format!("&{}{}*{snip}", mutability.prefix_str(), "*".repeat(indexed_ref_count))
                            };
                            (DEREF_BY_SLICING_LINT, "dereference the original value instead", sugg)
                        } else {
                            return;
                        }
                    } else {
                        return;
                    }
                } else {
                    return;
                };

                span_lint_and_sugg(
                    cx,
                    lint,
                    expr.span,
                    msg,
                    help,
                    sugg,
                    app,
                );
            }
        }
    }
}
