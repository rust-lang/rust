use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{differing_macro_contexts, eq_expr_value};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual swapping.
    ///
    /// ### Why is this bad?
    /// The `std::mem::swap` function exposes the intent better
    /// without deinitializing or copying either variable.
    ///
    /// ### Example
    /// ```rust
    /// let mut a = 42;
    /// let mut b = 1337;
    ///
    /// let t = b;
    /// b = a;
    /// a = t;
    /// ```
    /// Use std::mem::swap():
    /// ```rust
    /// let mut a = 1;
    /// let mut b = 2;
    /// std::mem::swap(&mut a, &mut b);
    /// ```
    pub MANUAL_SWAP,
    complexity,
    "manual swap of two variables"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `foo = bar; bar = foo` sequences.
    ///
    /// ### Why is this bad?
    /// This looks like a failed attempt to swap.
    ///
    /// ### Example
    /// ```rust
    /// # let mut a = 1;
    /// # let mut b = 2;
    /// a = b;
    /// b = a;
    /// ```
    /// If swapping is intended, use `swap()` instead:
    /// ```rust
    /// # let mut a = 1;
    /// # let mut b = 2;
    /// std::mem::swap(&mut a, &mut b);
    /// ```
    pub ALMOST_SWAPPED,
    correctness,
    "`foo = bar; bar = foo` sequence"
}

declare_clippy_lint! {
    /// **What it does:** Checks for uses of xor-based swaps.
    ///
    /// **Why is this bad?** The `std::mem::swap` function exposes the intent better
    /// without deinitializing or copying either variable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let mut a = 1;
    /// let mut b = 2;
    ///
    /// a ^= b;
    /// b ^= a;
    /// a ^= b;
    /// ```
    ///
    /// Use std::mem::swap() instead:
    /// ```rust
    /// let mut a = 1;
    /// let mut b = 2;
    /// std::mem::swap(&mut a, &mut b);
    /// ```
    pub XOR_SWAP,
    complexity,
    "xor-based swap of two variables"
}

declare_lint_pass!(Swap => [MANUAL_SWAP, ALMOST_SWAPPED, XOR_SWAP]);

impl<'tcx> LateLintPass<'tcx> for Swap {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        check_manual_swap(cx, block);
        check_suspicious_swap(cx, block);
        check_xor_swap(cx, block);
    }
}

/// Implementation of the `MANUAL_SWAP` lint.
fn check_manual_swap(cx: &LateContext<'_>, block: &Block<'_>) {
    for w in block.stmts.windows(3) {
        if_chain! {
            // let t = foo();
            if let StmtKind::Local(tmp) = w[0].kind;
            if let Some(tmp_init) = tmp.init;
            if let PatKind::Binding(.., ident, None) = tmp.pat.kind;

            // foo() = bar();
            if let StmtKind::Semi(first) = w[1].kind;
            if let ExprKind::Assign(lhs1, rhs1, _) = first.kind;

            // bar() = t;
            if let StmtKind::Semi(second) = w[2].kind;
            if let ExprKind::Assign(lhs2, rhs2, _) = second.kind;
            if let ExprKind::Path(QPath::Resolved(None, rhs2)) = rhs2.kind;
            if rhs2.segments.len() == 1;

            if ident.name == rhs2.segments[0].ident.name;
            if eq_expr_value(cx, tmp_init, lhs1);
            if eq_expr_value(cx, rhs1, lhs2);
            then {
                if let ExprKind::Field(lhs1, _) = lhs1.kind {
                    if let ExprKind::Field(lhs2, _) = lhs2.kind {
                        if lhs1.hir_id.owner == lhs2.hir_id.owner {
                            return;
                        }
                    }
                }

                let mut applicability = Applicability::MachineApplicable;

                let slice = check_for_slice(cx, lhs1, lhs2);
                let (replace, what, sugg) = if let Slice::NotSwappable = slice {
                    return;
                } else if let Slice::Swappable(slice, idx1, idx2) = slice {
                    if let Some(slice) = Sugg::hir_opt(cx, slice) {
                        (
                            false,
                            format!(" elements of `{}`", slice),
                            format!(
                                "{}.swap({}, {})",
                                slice.maybe_par(),
                                snippet_with_applicability(cx, idx1.span, "..", &mut applicability),
                                snippet_with_applicability(cx, idx2.span, "..", &mut applicability),
                            ),
                        )
                    } else {
                        (false, String::new(), String::new())
                    }
                } else if let (Some(first), Some(second)) = (Sugg::hir_opt(cx, lhs1), Sugg::hir_opt(cx, rhs1)) {
                    (
                        true,
                        format!(" `{}` and `{}`", first, second),
                        format!("std::mem::swap({}, {})", first.mut_addr(), second.mut_addr()),
                    )
                } else {
                    (true, String::new(), String::new())
                };

                let span = w[0].span.to(second.span);

                span_lint_and_then(
                    cx,
                    MANUAL_SWAP,
                    span,
                    &format!("this looks like you are swapping{} manually", what),
                    |diag| {
                        if !sugg.is_empty() {
                            diag.span_suggestion(
                                span,
                                "try",
                                sugg,
                                applicability,
                            );

                            if replace {
                                diag.note("or maybe you should use `std::mem::replace`?");
                            }
                        }
                    }
                );
            }
        }
    }
}

enum Slice<'a> {
    /// `slice.swap(idx1, idx2)` can be used
    ///
    /// ## Example
    ///
    /// ```rust
    /// # let mut a = vec![0, 1];
    /// let t = a[1];
    /// a[1] = a[0];
    /// a[0] = t;
    /// // can be written as
    /// a.swap(0, 1);
    /// ```
    Swappable(&'a Expr<'a>, &'a Expr<'a>, &'a Expr<'a>),
    /// The `swap` function cannot be used.
    ///
    /// ## Example
    ///
    /// ```rust
    /// # let mut a = [vec![1, 2], vec![3, 4]];
    /// let t = a[0][1];
    /// a[0][1] = a[1][0];
    /// a[1][0] = t;
    /// ```
    NotSwappable,
    /// Not a slice
    None,
}

/// Checks if both expressions are index operations into "slice-like" types.
fn check_for_slice<'a>(cx: &LateContext<'_>, lhs1: &'a Expr<'_>, lhs2: &'a Expr<'_>) -> Slice<'a> {
    if let ExprKind::Index(lhs1, idx1) = lhs1.kind {
        if let ExprKind::Index(lhs2, idx2) = lhs2.kind {
            if eq_expr_value(cx, lhs1, lhs2) {
                let ty = cx.typeck_results().expr_ty(lhs1).peel_refs();

                if matches!(ty.kind(), ty::Slice(_))
                    || matches!(ty.kind(), ty::Array(_, _))
                    || is_type_diagnostic_item(cx, ty, sym::vec_type)
                    || is_type_diagnostic_item(cx, ty, sym::vecdeque_type)
                {
                    return Slice::Swappable(lhs1, idx1, idx2);
                }
            } else {
                return Slice::NotSwappable;
            }
        }
    }

    Slice::None
}

/// Implementation of the `ALMOST_SWAPPED` lint.
fn check_suspicious_swap(cx: &LateContext<'_>, block: &Block<'_>) {
    for w in block.stmts.windows(2) {
        if_chain! {
            if let StmtKind::Semi(first) = w[0].kind;
            if let StmtKind::Semi(second) = w[1].kind;
            if !differing_macro_contexts(first.span, second.span);
            if let ExprKind::Assign(lhs0, rhs0, _) = first.kind;
            if let ExprKind::Assign(lhs1, rhs1, _) = second.kind;
            if eq_expr_value(cx, lhs0, rhs1);
            if eq_expr_value(cx, lhs1, rhs0);
            then {
                let lhs0 = Sugg::hir_opt(cx, lhs0);
                let rhs0 = Sugg::hir_opt(cx, rhs0);
                let (what, lhs, rhs) = if let (Some(first), Some(second)) = (lhs0, rhs0) {
                    (
                        format!(" `{}` and `{}`", first, second),
                        first.mut_addr().to_string(),
                        second.mut_addr().to_string(),
                    )
                } else {
                    (String::new(), String::new(), String::new())
                };

                let span = first.span.to(second.span);

                span_lint_and_then(cx,
                                   ALMOST_SWAPPED,
                                   span,
                                   &format!("this looks like you are trying to swap{}", what),
                                   |diag| {
                                       if !what.is_empty() {
                                           diag.span_suggestion(
                                               span,
                                               "try",
                                               format!(
                                                   "std::mem::swap({}, {})",
                                                   lhs,
                                                   rhs,
                                               ),
                                               Applicability::MaybeIncorrect,
                                           );
                                           diag.note("or maybe you should use `std::mem::replace`?");
                                       }
                                   });
            }
        }
    }
}

/// Implementation of the `XOR_SWAP` lint.
fn check_xor_swap(cx: &LateContext<'_>, block: &Block<'_>) {
    for window in block.stmts.windows(3) {
        if_chain! {
            if let Some((lhs0, rhs0)) = extract_sides_of_xor_assign(&window[0]);
            if let Some((lhs1, rhs1)) = extract_sides_of_xor_assign(&window[1]);
            if let Some((lhs2, rhs2)) = extract_sides_of_xor_assign(&window[2]);
            if eq_expr_value(cx, lhs0, rhs1)
               && eq_expr_value(cx, rhs1, lhs2)
               && eq_expr_value(cx, rhs0, lhs1)
               && eq_expr_value(cx, lhs1, rhs2);
            then {
                let span = window[0].span.to(window[2].span);
                let mut applicability = Applicability::MachineApplicable;
                match check_for_slice(cx, lhs0, rhs0) {
                    Slice::Swappable(slice, idx0, idx1) => {
                        if let Some(slice) = Sugg::hir_opt(cx, slice) {
                            span_lint_and_sugg(
                                cx,
                                XOR_SWAP,
                                span,
                                &format!(
                                    "this xor-based swap of the elements of `{}` can be \
                                    more clearly expressed using `.swap`",
                                    slice
                                ),
                                "try",
                                format!(
                                    "{}.swap({}, {})",
                                    slice.maybe_par(),
                                    snippet_with_applicability(cx, idx0.span, "..", &mut applicability),
                                    snippet_with_applicability(cx, idx1.span, "..", &mut applicability)
                                ),
                                applicability
                            )
                        }
                    }
                    Slice::None => {
                        if let (Some(first), Some(second)) = (Sugg::hir_opt(cx, lhs0), Sugg::hir_opt(cx, rhs0)) {
                            span_lint_and_sugg(
                                cx,
                                XOR_SWAP,
                                span,
                                &format!(
                                    "this xor-based swap of `{}` and `{}` can be \
                                    more clearly expressed using `std::mem::swap`",
                                    first, second
                                ),
                                "try",
                                format!("std::mem::swap({}, {})", first.mut_addr(), second.mut_addr()),
                                applicability,
                            );
                        }
                    }
                    Slice::NotSwappable => {}
                }
            }
        };
    }
}

/// Returns the lhs and rhs of an xor assignment statement.  
fn extract_sides_of_xor_assign<'a, 'hir>(stmt: &'a Stmt<'hir>) -> Option<(&'a Expr<'hir>, &'a Expr<'hir>)> {
    if let StmtKind::Semi(expr) = stmt.kind {
        if let ExprKind::AssignOp(
            Spanned {
                node: BinOpKind::BitXor,
                ..
            },
            lhs,
            rhs,
        ) = expr.kind
        {
            return Some((lhs, rhs));
        }
    }
    None
}
