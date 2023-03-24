use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::snippet_with_context;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{can_mut_borrow_both, eq_expr_value, in_constant, std_or_core};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::SyntaxContext;
use rustc_span::{sym, symbol::Ident, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual swapping.
    ///
    /// Note that the lint will not be emitted in const blocks, as the suggestion would not be applicable.
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
    #[clippy::version = "pre 1.29.0"]
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
    #[clippy::version = "pre 1.29.0"]
    pub ALMOST_SWAPPED,
    correctness,
    "`foo = bar; bar = foo` sequence"
}

declare_lint_pass!(Swap => [MANUAL_SWAP, ALMOST_SWAPPED]);

impl<'tcx> LateLintPass<'tcx> for Swap {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        check_manual_swap(cx, block);
        check_suspicious_swap(cx, block);
        check_xor_swap(cx, block);
    }
}

fn generate_swap_warning(cx: &LateContext<'_>, e1: &Expr<'_>, e2: &Expr<'_>, span: Span, is_xor_based: bool) {
    let ctxt = span.ctxt();
    let mut applicability = Applicability::MachineApplicable;

    if !can_mut_borrow_both(cx, e1, e2) {
        if let ExprKind::Index(lhs1, idx1) = e1.kind
            && let ExprKind::Index(lhs2, idx2) = e2.kind
            && eq_expr_value(cx, lhs1, lhs2)
            && e1.span.ctxt() == ctxt
            && e2.span.ctxt() == ctxt
        {
            let ty = cx.typeck_results().expr_ty(lhs1).peel_refs();

            if matches!(ty.kind(), ty::Slice(_))
                || matches!(ty.kind(), ty::Array(_, _))
                || is_type_diagnostic_item(cx, ty, sym::Vec)
                || is_type_diagnostic_item(cx, ty, sym::VecDeque)
            {
                let slice = Sugg::hir_with_applicability(cx, lhs1, "<slice>", &mut applicability);
                span_lint_and_sugg(
                    cx,
                    MANUAL_SWAP,
                    span,
                    &format!("this looks like you are swapping elements of `{slice}` manually"),
                    "try",
                    format!(
                        "{}.swap({}, {});",
                        slice.maybe_par(),
                        snippet_with_context(cx, idx1.span, ctxt, "..", &mut applicability).0,
                        snippet_with_context(cx, idx2.span, ctxt, "..", &mut applicability).0,
                    ),
                    applicability,
                );
            }
        }
        return;
    }

    let first = Sugg::hir_with_context(cx, e1, ctxt, "..", &mut applicability);
    let second = Sugg::hir_with_context(cx, e2, ctxt, "..", &mut applicability);
    let Some(sugg) = std_or_core(cx) else { return };

    span_lint_and_then(
        cx,
        MANUAL_SWAP,
        span,
        &format!("this looks like you are swapping `{first}` and `{second}` manually"),
        |diag| {
            diag.span_suggestion(
                span,
                "try",
                format!("{sugg}::mem::swap({}, {});", first.mut_addr(), second.mut_addr()),
                applicability,
            );
            if !is_xor_based {
                diag.note(format!("or maybe you should use `{sugg}::mem::replace`?"));
            }
        },
    );
}

/// Implementation of the `MANUAL_SWAP` lint.
fn check_manual_swap(cx: &LateContext<'_>, block: &Block<'_>) {
    if in_constant(cx, block.hir_id) {
        return;
    }

    for [s1, s2, s3] in block.stmts.array_windows::<3>() {
        if_chain! {
            // let t = foo();
            if let StmtKind::Local(tmp) = s1.kind;
            if let Some(tmp_init) = tmp.init;
            if let PatKind::Binding(.., ident, None) = tmp.pat.kind;

            // foo() = bar();
            if let StmtKind::Semi(first) = s2.kind;
            if let ExprKind::Assign(lhs1, rhs1, _) = first.kind;

            // bar() = t;
            if let StmtKind::Semi(second) = s3.kind;
            if let ExprKind::Assign(lhs2, rhs2, _) = second.kind;
            if let ExprKind::Path(QPath::Resolved(None, rhs2)) = rhs2.kind;
            if rhs2.segments.len() == 1;

            if ident.name == rhs2.segments[0].ident.name;
            if eq_expr_value(cx, tmp_init, lhs1);
            if eq_expr_value(cx, rhs1, lhs2);

            let ctxt = s1.span.ctxt();
            if s2.span.ctxt() == ctxt;
            if s3.span.ctxt() == ctxt;
            if first.span.ctxt() == ctxt;
            if second.span.ctxt() == ctxt;

            then {
                let span = s1.span.to(s3.span);
                generate_swap_warning(cx, lhs1, lhs2, span, false);
            }
        }
    }
}

/// Implementation of the `ALMOST_SWAPPED` lint.
fn check_suspicious_swap(cx: &LateContext<'_>, block: &Block<'_>) {
    for [first, second] in block.stmts.array_windows() {
        if let Some((lhs0, rhs0)) = parse(first)
            && let Some((lhs1, rhs1)) = parse(second)
            && first.span.eq_ctxt(second.span)
			&& !in_external_macro(cx.sess(), first.span)
            && is_same(cx, lhs0, rhs1)
            && is_same(cx, lhs1, rhs0)
			&& !is_same(cx, lhs1, rhs1) // Ignore a = b; a = a (#10421)
            && let Some(lhs_sugg) = match &lhs0 {
                ExprOrIdent::Expr(expr) => Sugg::hir_opt(cx, expr),
                ExprOrIdent::Ident(ident) => Some(Sugg::NonParen(ident.as_str().into())),
            }
            && let Some(rhs_sugg) = Sugg::hir_opt(cx, rhs0)
        {
            let span = first.span.to(rhs1.span);
            let Some(sugg) = std_or_core(cx) else { return };
            span_lint_and_then(
                cx,
                ALMOST_SWAPPED,
                span,
                &format!("this looks like you are trying to swap `{lhs_sugg}` and `{rhs_sugg}`"),
                |diag| {
                    diag.span_suggestion(
                        span,
                        "try",
                        format!("{sugg}::mem::swap({}, {})", lhs_sugg.mut_addr(), rhs_sugg.mut_addr()),
                        Applicability::MaybeIncorrect,
                    );
                    diag.note(format!("or maybe you should use `{sugg}::mem::replace`?"));
                },
            );
        }
    }
}

fn is_same(cx: &LateContext<'_>, lhs: ExprOrIdent<'_>, rhs: &Expr<'_>) -> bool {
    match lhs {
        ExprOrIdent::Expr(expr) => eq_expr_value(cx, expr, rhs),
        ExprOrIdent::Ident(ident) => {
            if let ExprKind::Path(QPath::Resolved(None, path)) = rhs.kind
                && let [segment] = &path.segments
                && segment.ident == ident
            {
                true
            } else {
                false
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ExprOrIdent<'a> {
    Expr(&'a Expr<'a>),
    Ident(Ident),
}

fn parse<'a, 'hir>(stmt: &'a Stmt<'hir>) -> Option<(ExprOrIdent<'hir>, &'a Expr<'hir>)> {
    if let StmtKind::Semi(expr) = stmt.kind {
        if let ExprKind::Assign(lhs, rhs, _) = expr.kind {
            return Some((ExprOrIdent::Expr(lhs), rhs));
        }
    } else if let StmtKind::Local(expr) = stmt.kind {
        if let Some(rhs) = expr.init {
            if let PatKind::Binding(_, _, ident_l, _) = expr.pat.kind {
                return Some((ExprOrIdent::Ident(ident_l), rhs));
            }
        }
    }
    None
}

/// Implementation of the xor case for `MANUAL_SWAP` lint.
fn check_xor_swap(cx: &LateContext<'_>, block: &Block<'_>) {
    for [s1, s2, s3] in block.stmts.array_windows::<3>() {
        let ctxt = s1.span.ctxt();
        if_chain! {
            if let Some((lhs0, rhs0)) = extract_sides_of_xor_assign(s1, ctxt);
            if let Some((lhs1, rhs1)) = extract_sides_of_xor_assign(s2, ctxt);
            if let Some((lhs2, rhs2)) = extract_sides_of_xor_assign(s3, ctxt);
            if eq_expr_value(cx, lhs0, rhs1);
            if eq_expr_value(cx, lhs2, rhs1);
            if eq_expr_value(cx, lhs1, rhs0);
            if eq_expr_value(cx, lhs1, rhs2);
            if s2.span.ctxt() == ctxt;
            if s3.span.ctxt() == ctxt;
            then {
                let span = s1.span.to(s3.span);
                generate_swap_warning(cx, lhs0, rhs0, span, true);
            }
        };
    }
}

/// Returns the lhs and rhs of an xor assignment statement.
fn extract_sides_of_xor_assign<'a, 'hir>(
    stmt: &'a Stmt<'hir>,
    ctxt: SyntaxContext,
) -> Option<(&'a Expr<'hir>, &'a Expr<'hir>)> {
    if let StmtKind::Semi(expr) = stmt.kind
        && let ExprKind::AssignOp(
            Spanned {
                node: BinOpKind::BitXor,
                ..
            },
            lhs,
            rhs,
        ) = expr.kind
        && expr.span.ctxt() == ctxt
    {
        Some((lhs, rhs))
    } else {
        None
    }
}
