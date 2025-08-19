use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::{VecInitKind, get_vec_init_kind};
use clippy_utils::source::snippet;
use clippy_utils::{is_from_proc_macro, path_to_local_id, sym};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{BindingMode, Block, Expr, ExprKind, HirId, LetStmt, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Informs the user about a more concise way to create a vector with a known capacity.
    ///
    /// ### Why is this bad?
    /// The `Vec::with_capacity` constructor is less complex.
    ///
    /// ### Example
    /// ```no_run
    /// let mut v: Vec<usize> = vec![];
    /// v.reserve(10);
    /// ```
    /// Use instead:
    /// ```no_run
    /// let mut v: Vec<usize> = Vec::with_capacity(10);
    /// ```
    #[clippy::version = "1.74.0"]
    pub RESERVE_AFTER_INITIALIZATION,
    complexity,
    "`reserve` called immediately after `Vec` creation"
}
impl_lint_pass!(ReserveAfterInitialization => [RESERVE_AFTER_INITIALIZATION]);

#[derive(Default)]
pub struct ReserveAfterInitialization {
    searcher: Option<VecReserveSearcher>,
}

struct VecReserveSearcher {
    local_id: HirId,
    err_span: Span,
    init_part: String,
    space_hint: String,
}
impl VecReserveSearcher {
    fn display_err(&self, cx: &LateContext<'_>) {
        if self.space_hint.is_empty() {
            return;
        }

        let s = format!("{}Vec::with_capacity({});", self.init_part, self.space_hint);

        span_lint_and_sugg(
            cx,
            RESERVE_AFTER_INITIALIZATION,
            self.err_span,
            "call to `reserve` immediately after creation",
            "consider using `Vec::with_capacity(/* Space hint */)`",
            s,
            Applicability::HasPlaceholders,
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for ReserveAfterInitialization {
    fn check_block(&mut self, _: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        self.searcher = None;
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx LetStmt<'tcx>) {
        if let Some(init_expr) = local.init
            && let PatKind::Binding(BindingMode::MUT, id, _, None) = local.pat.kind
            && !local.span.in_external_macro(cx.sess().source_map())
            && let Some(init) = get_vec_init_kind(cx, init_expr)
            && !matches!(
                init,
                VecInitKind::WithExprCapacity(_) | VecInitKind::WithConstCapacity(_)
            )
        {
            self.searcher = Some(VecReserveSearcher {
                local_id: id,
                err_span: local.span,
                init_part: snippet(
                    cx,
                    local
                        .span
                        .shrink_to_lo()
                        .to(init_expr.span.source_callsite().shrink_to_lo()),
                    "..",
                )
                .into_owned(),
                space_hint: String::new(),
            });
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if self.searcher.is_none()
            && let ExprKind::Assign(left, right, _) = expr.kind
            && let ExprKind::Path(QPath::Resolved(None, path)) = left.kind
            && let Res::Local(id) = path.res
            && !expr.span.in_external_macro(cx.sess().source_map())
            && let Some(init) = get_vec_init_kind(cx, right)
            && !matches!(
                init,
                VecInitKind::WithExprCapacity(_) | VecInitKind::WithConstCapacity(_)
            )
        {
            self.searcher = Some(VecReserveSearcher {
                local_id: id,
                err_span: expr.span,
                init_part: snippet(
                    cx,
                    left.span.shrink_to_lo().to(right.span.source_callsite().shrink_to_lo()),
                    "..",
                )
                .into_owned(), // see `assign_expression` test
                space_hint: String::new(),
            });
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let Some(searcher) = self.searcher.take() {
            if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = stmt.kind
                && let ExprKind::MethodCall(name, self_arg, [space_hint], _) = expr.kind
                && path_to_local_id(self_arg, searcher.local_id)
                && name.ident.name == sym::reserve
                && !is_from_proc_macro(cx, expr)
            {
                self.searcher = Some(VecReserveSearcher {
                    err_span: searcher.err_span.to(stmt.span),
                    space_hint: snippet(cx, space_hint.span, "..").into_owned(),
                    ..searcher
                });
            } else {
                searcher.display_err(cx);
            }
        }
    }

    fn check_block_post(&mut self, cx: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        if let Some(searcher) = self.searcher.take() {
            searcher.display_err(cx);
        }
    }
}
