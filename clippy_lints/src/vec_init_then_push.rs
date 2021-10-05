use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::{get_vec_init_kind, VecInitKind};
use clippy_utils::source::snippet;
use clippy_utils::{path_to_local, path_to_local_id};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, Block, Expr, ExprKind, HirId, Local, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `push` immediately after creating a new `Vec`.
    ///
    /// ### Why is this bad?
    /// The `vec![]` macro is both more performant and easier to read than
    /// multiple `push` calls.
    ///
    /// ### Example
    /// ```rust
    /// let mut v = Vec::new();
    /// v.push(0);
    /// ```
    /// Use instead:
    /// ```rust
    /// let v = vec![0];
    /// ```
    pub VEC_INIT_THEN_PUSH,
    perf,
    "`push` immediately after `Vec` creation"
}

impl_lint_pass!(VecInitThenPush => [VEC_INIT_THEN_PUSH]);

#[derive(Default)]
pub struct VecInitThenPush {
    searcher: Option<VecPushSearcher>,
}

struct VecPushSearcher {
    local_id: HirId,
    init: VecInitKind,
    lhs_is_local: bool,
    lhs_span: Span,
    err_span: Span,
    found: u64,
}
impl VecPushSearcher {
    fn display_err(&self, cx: &LateContext<'_>) {
        match self.init {
            _ if self.found == 0 => return,
            VecInitKind::WithLiteralCapacity(x) if x > self.found => return,
            VecInitKind::WithExprCapacity(_) => return,
            _ => (),
        };

        let mut s = if self.lhs_is_local {
            String::from("let ")
        } else {
            String::new()
        };
        s.push_str(&snippet(cx, self.lhs_span, ".."));
        s.push_str(" = vec![..];");

        span_lint_and_sugg(
            cx,
            VEC_INIT_THEN_PUSH,
            self.err_span,
            "calls to `push` immediately after creation",
            "consider using the `vec![]` macro",
            s,
            Applicability::HasPlaceholders,
        );
    }
}

impl LateLintPass<'_> for VecInitThenPush {
    fn check_block(&mut self, _: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        self.searcher = None;
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'tcx>) {
        if_chain! {
            if !in_external_macro(cx.sess(), local.span);
            if let Some(init) = local.init;
            if let PatKind::Binding(BindingAnnotation::Mutable, id, _, None) = local.pat.kind;
            if let Some(init_kind) = get_vec_init_kind(cx, init);
            then {
                self.searcher = Some(VecPushSearcher {
                        local_id: id,
                        init: init_kind,
                        lhs_is_local: true,
                        lhs_span: local.ty.map_or(local.pat.span, |t| local.pat.span.to(t.span)),
                        err_span: local.span,
                        found: 0,
                    });
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if self.searcher.is_none();
            if !in_external_macro(cx.sess(), expr.span);
            if let ExprKind::Assign(left, right, _) = expr.kind;
            if let Some(id) = path_to_local(left);
            if let Some(init_kind) = get_vec_init_kind(cx, right);
            then {
                self.searcher = Some(VecPushSearcher {
                    local_id: id,
                    init: init_kind,
                    lhs_is_local: false,
                    lhs_span: left.span,
                    err_span: expr.span,
                    found: 0,
                });
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let Some(searcher) = self.searcher.take() {
            if_chain! {
                if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = stmt.kind;
                if let ExprKind::MethodCall(path, _, [self_arg, _], _) = expr.kind;
                if path_to_local_id(self_arg, searcher.local_id);
                if path.ident.name.as_str() == "push";
                then {
                    self.searcher = Some(VecPushSearcher {
                        found: searcher.found + 1,
                        err_span: searcher.err_span.to(stmt.span),
                        .. searcher
                    });
                } else {
                    searcher.display_err(cx);
                }
            }
        }
    }

    fn check_block_post(&mut self, cx: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        if let Some(searcher) = self.searcher.take() {
            searcher.display_err(cx);
        }
    }
}
