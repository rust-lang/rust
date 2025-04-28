use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet_indent, snippet_with_context};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;

use clippy_utils::{can_mut_borrow_both, eq_expr_value, is_in_const_context, std_or_core};
use itertools::Itertools;

use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::intravisit::{Visitor, walk_expr};

use rustc_errors::Applicability;
use rustc_hir::{AssignOpKind, Block, Expr, ExprKind, LetStmt, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;
use rustc_span::{Span, SyntaxContext, sym};

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
    /// ```no_run
    /// let mut a = 42;
    /// let mut b = 1337;
    ///
    /// let t = b;
    /// b = a;
    /// a = t;
    /// ```
    /// Use std::mem::swap():
    /// ```no_run
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
    /// ```no_run
    /// # let mut a = 1;
    /// # let mut b = 2;
    /// a = b;
    /// b = a;
    /// ```
    /// If swapping is intended, use `swap()` instead:
    /// ```no_run
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

#[allow(clippy::too_many_arguments)]
fn generate_swap_warning<'tcx>(
    block: &'tcx Block<'tcx>,
    cx: &LateContext<'tcx>,
    e1: &'tcx Expr<'tcx>,
    e2: &'tcx Expr<'tcx>,
    rhs1: &'tcx Expr<'tcx>,
    rhs2: &'tcx Expr<'tcx>,
    span: Span,
    is_xor_based: bool,
) {
    let ctxt = span.ctxt();
    let mut applicability = Applicability::MachineApplicable;

    if !can_mut_borrow_both(cx, e1, e2) {
        if let ExprKind::Index(lhs1, idx1, _) = e1.kind
            && let ExprKind::Index(lhs2, idx2, _) = e2.kind
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
                    format!("this looks like you are swapping elements of `{slice}` manually"),
                    "try",
                    format!(
                        "{}{}.swap({}, {});",
                        IndexBinding {
                            block,
                            swap1_idx: idx1,
                            swap2_idx: idx2,
                            suggest_span: span,
                            cx,
                            ctxt,
                            applicability: &mut applicability,
                        }
                        .snippet_index_bindings(&[idx1, idx2, rhs1, rhs2]),
                        slice.maybe_paren(),
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
        format!("this looks like you are swapping `{first}` and `{second}` manually"),
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
fn check_manual_swap<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
    if is_in_const_context(cx) {
        return;
    }

    for [s1, s2, s3] in block.stmts.array_windows::<3>() {
        if let StmtKind::Let(tmp) = s1.kind
            // let t = foo();
            && let Some(tmp_init) = tmp.init
            && let PatKind::Binding(.., ident, None) = tmp.pat.kind

            // foo() = bar();
            && let StmtKind::Semi(first) = s2.kind
            && let ExprKind::Assign(lhs1, rhs1, _) = first.kind

            // bar() = t;
            && let StmtKind::Semi(second) = s3.kind
            && let ExprKind::Assign(lhs2, rhs2, _) = second.kind
            && let ExprKind::Path(QPath::Resolved(None, rhs2_path)) = rhs2.kind
            && rhs2_path.segments.len() == 1

            && ident.name == rhs2_path.segments[0].ident.name
            && eq_expr_value(cx, tmp_init, lhs1)
            && eq_expr_value(cx, rhs1, lhs2)

            && let ctxt = s1.span.ctxt()
            && s2.span.ctxt() == ctxt
            && s3.span.ctxt() == ctxt
            && first.span.ctxt() == ctxt
            && second.span.ctxt() == ctxt
        {
            let span = s1.span.to(s3.span);
            generate_swap_warning(block, cx, lhs1, lhs2, rhs1, rhs2, span, false);
        }
    }
}

/// Implementation of the `ALMOST_SWAPPED` lint.
fn check_suspicious_swap(cx: &LateContext<'_>, block: &Block<'_>) {
    for [first, second] in block.stmts.array_windows() {
        if let Some((lhs0, rhs0)) = parse(first)
            && let Some((lhs1, rhs1)) = parse(second)
            && first.span.eq_ctxt(second.span)
			&& !first.span.in_external_macro(cx.sess().source_map())
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
                format!("this looks like you are trying to swap `{lhs_sugg}` and `{rhs_sugg}`"),
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
        },
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
    } else if let StmtKind::Let(expr) = stmt.kind
        && let Some(rhs) = expr.init
        && let PatKind::Binding(_, _, ident_l, _) = expr.pat.kind
    {
        return Some((ExprOrIdent::Ident(ident_l), rhs));
    }
    None
}

/// Implementation of the xor case for `MANUAL_SWAP` lint.
fn check_xor_swap<'tcx>(cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
    for [s1, s2, s3] in block.stmts.array_windows::<3>() {
        let ctxt = s1.span.ctxt();
        if let Some((lhs0, rhs0)) = extract_sides_of_xor_assign(s1, ctxt)
            && let Some((lhs1, rhs1)) = extract_sides_of_xor_assign(s2, ctxt)
            && let Some((lhs2, rhs2)) = extract_sides_of_xor_assign(s3, ctxt)
            && eq_expr_value(cx, lhs0, rhs1)
            && eq_expr_value(cx, lhs2, rhs1)
            && eq_expr_value(cx, lhs1, rhs0)
            && eq_expr_value(cx, lhs1, rhs2)
            && s2.span.ctxt() == ctxt
            && s3.span.ctxt() == ctxt
        {
            let span = s1.span.to(s3.span);
            generate_swap_warning(block, cx, lhs0, rhs0, rhs1, rhs2, span, true);
        }
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
                node: AssignOpKind::BitXorAssign,
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

struct IndexBinding<'a, 'tcx> {
    block: &'a Block<'a>,
    swap1_idx: &'a Expr<'a>,
    swap2_idx: &'a Expr<'a>,
    suggest_span: Span,
    cx: &'a LateContext<'tcx>,
    ctxt: SyntaxContext,
    applicability: &'a mut Applicability,
}

impl<'tcx> IndexBinding<'_, 'tcx> {
    fn snippet_index_bindings(&mut self, exprs: &[&'tcx Expr<'tcx>]) -> String {
        let mut bindings = FxIndexSet::default();
        for expr in exprs {
            bindings.insert(self.snippet_index_binding(expr));
        }
        bindings.into_iter().join("")
    }

    fn snippet_index_binding(&mut self, expr: &'tcx Expr<'tcx>) -> String {
        match expr.kind {
            ExprKind::Binary(_, lhs, rhs) => {
                if matches!(lhs.kind, ExprKind::Lit(_)) && matches!(rhs.kind, ExprKind::Lit(_)) {
                    return String::new();
                }
                let lhs_snippet = self.snippet_index_binding(lhs);
                let rhs_snippet = self.snippet_index_binding(rhs);
                format!("{lhs_snippet}{rhs_snippet}")
            },
            ExprKind::Path(QPath::Resolved(_, path)) => {
                let init = self.cx.expr_or_init(expr);

                let Some(first_segment) = path.segments.first() else {
                    return String::new();
                };
                if !self.suggest_span.contains(init.span) || !self.is_used_other_than_swapping(first_segment.ident) {
                    return String::new();
                }

                let init_str = snippet_with_context(self.cx, init.span, self.ctxt, "", self.applicability)
                    .0
                    .to_string();
                let indent_str = snippet_indent(self.cx, init.span);
                let indent_str = indent_str.as_deref().unwrap_or("");

                format!("let {} = {init_str};\n{indent_str}", first_segment.ident)
            },
            _ => String::new(),
        }
    }

    fn is_used_other_than_swapping(&mut self, idx_ident: Ident) -> bool {
        if Self::is_used_slice_indexed(self.swap1_idx, idx_ident)
            || Self::is_used_slice_indexed(self.swap2_idx, idx_ident)
        {
            return true;
        }
        self.is_used_after_swap(idx_ident)
    }

    fn is_used_after_swap(&mut self, idx_ident: Ident) -> bool {
        let mut v = IndexBindingVisitor {
            idx: idx_ident,
            suggest_span: self.suggest_span,
            found_used: false,
        };

        for stmt in self.block.stmts {
            match stmt.kind {
                StmtKind::Expr(expr) | StmtKind::Semi(expr) => v.visit_expr(expr),
                StmtKind::Let(LetStmt { init, .. }) => {
                    if let Some(init) = init.as_ref() {
                        v.visit_expr(init);
                    }
                },
                StmtKind::Item(_) => {},
            }
        }

        v.found_used
    }

    fn is_used_slice_indexed(swap_index: &Expr<'_>, idx_ident: Ident) -> bool {
        match swap_index.kind {
            ExprKind::Binary(_, lhs, rhs) => {
                if matches!(lhs.kind, ExprKind::Lit(_)) && matches!(rhs.kind, ExprKind::Lit(_)) {
                    return false;
                }
                Self::is_used_slice_indexed(lhs, idx_ident) || Self::is_used_slice_indexed(rhs, idx_ident)
            },
            ExprKind::Path(QPath::Resolved(_, path)) => path.segments.first().is_some_and(|idx| idx.ident == idx_ident),
            _ => false,
        }
    }
}

struct IndexBindingVisitor {
    idx: Ident,
    suggest_span: Span,
    found_used: bool,
}

impl<'tcx> Visitor<'tcx> for IndexBindingVisitor {
    fn visit_path_segment(&mut self, path_segment: &'tcx rustc_hir::PathSegment<'tcx>) -> Self::Result {
        if path_segment.ident == self.idx {
            self.found_used = true;
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> Self::Result {
        if expr.span.hi() <= self.suggest_span.hi() {
            return;
        }

        match expr.kind {
            ExprKind::Path(QPath::Resolved(_, path)) => {
                for segment in path.segments {
                    self.visit_path_segment(segment);
                }
            },
            _ => walk_expr(self, expr),
        }
    }
}
