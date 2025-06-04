use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{SpanRangeExt, snippet};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{path_to_local_id, sym};
use rustc_ast::{LitKind, StrStyle};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{BindingMode, Block, Expr, ExprKind, HirId, LetStmt, PatKind, QPath, Stmt, StmtKind, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `push` immediately after creating a new `PathBuf`.
    ///
    /// ### Why is this bad?
    /// Multiple `.join()` calls are usually easier to read than multiple `.push`
    /// calls across multiple statements. It might also be possible to use
    /// `PathBuf::from` instead.
    ///
    /// ### Known problems
    /// `.join()` introduces an implicit `clone()`. `PathBuf::from` can alternatively be
    /// used when the `PathBuf` is newly constructed. This will avoid the implicit clone.
    ///
    /// ### Example
    /// ```rust
    /// # use std::path::PathBuf;
    /// let mut path_buf = PathBuf::new();
    /// path_buf.push("foo");
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::path::PathBuf;
    /// let path_buf = PathBuf::from("foo");
    /// // or
    /// let path_buf = PathBuf::new().join("foo");
    /// ```
    #[clippy::version = "1.82.0"]
    pub PATHBUF_INIT_THEN_PUSH,
    restriction,
    "`push` immediately after `PathBuf` creation"
}

impl_lint_pass!(PathbufThenPush<'_> => [PATHBUF_INIT_THEN_PUSH]);

#[derive(Default)]
pub struct PathbufThenPush<'tcx> {
    searcher: Option<PathbufPushSearcher<'tcx>>,
}

struct PathbufPushSearcher<'tcx> {
    local_id: HirId,
    lhs_is_let: bool,
    let_ty_span: Option<Span>,
    init_val: Expr<'tcx>,
    arg: Option<Expr<'tcx>>,
    name: Symbol,
    err_span: Span,
}

impl PathbufPushSearcher<'_> {
    /// Try to generate a suggestion with `PathBuf::from`.
    /// Returns `None` if the suggestion would be invalid.
    fn gen_pathbuf_from(&self, cx: &LateContext<'_>) -> Option<String> {
        if let ExprKind::Call(iter_expr, []) = &self.init_val.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, segment)) = &iter_expr.kind
            && let TyKind::Path(ty_path) = &ty.kind
            && let QPath::Resolved(None, path) = ty_path
            && let Res::Def(_, def_id) = &path.res
            && cx.tcx.is_diagnostic_item(sym::PathBuf, *def_id)
            && segment.ident.name == sym::new
            && let Some(arg) = self.arg
            && let ExprKind::Lit(x) = arg.kind
            && let LitKind::Str(_, StrStyle::Cooked) = x.node
            && let Some(s) = arg.span.get_source_text(cx)
        {
            Some(format!(" = PathBuf::from({s});"))
        } else {
            None
        }
    }

    fn gen_pathbuf_join(&self, cx: &LateContext<'_>) -> Option<String> {
        let arg = self.arg?;
        let arg_str = arg.span.get_source_text(cx)?;
        let init_val = self.init_val.span.get_source_text(cx)?;
        Some(format!(" = {init_val}.join({arg_str});"))
    }

    fn display_err(&self, cx: &LateContext<'_>) {
        if clippy_utils::attrs::span_contains_cfg(cx, self.err_span) {
            return;
        }
        let mut sugg = if self.lhs_is_let {
            String::from("let mut ")
        } else {
            String::new()
        };
        sugg.push_str(self.name.as_str());
        if let Some(span) = self.let_ty_span {
            sugg.push_str(": ");
            sugg.push_str(&snippet(cx, span, "_"));
        }
        match self.gen_pathbuf_from(cx) {
            Some(value) => {
                sugg.push_str(&value);
            },
            None => {
                if let Some(value) = self.gen_pathbuf_join(cx) {
                    sugg.push_str(&value);
                } else {
                    return;
                }
            },
        }

        span_lint_and_sugg(
            cx,
            PATHBUF_INIT_THEN_PUSH,
            self.err_span,
            "calls to `push` immediately after creation",
            "consider using the `.join()`",
            sugg,
            Applicability::HasPlaceholders,
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for PathbufThenPush<'tcx> {
    fn check_block(&mut self, _: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        self.searcher = None;
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx LetStmt<'tcx>) {
        if let Some(init_expr) = local.init
            && let PatKind::Binding(BindingMode::MUT, id, name, None) = local.pat.kind
            && !local.span.in_external_macro(cx.sess().source_map())
            && let ty = cx.typeck_results().pat_ty(local.pat)
            && is_type_diagnostic_item(cx, ty, sym::PathBuf)
        {
            self.searcher = Some(PathbufPushSearcher {
                local_id: id,
                lhs_is_let: true,
                let_ty_span: local.ty.map(|ty| ty.span),
                init_val: *init_expr,
                arg: None,
                name: name.name,
                err_span: local.span,
            });
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Assign(left, right, _) = expr.kind
            && let ExprKind::Path(QPath::Resolved(None, path)) = left.kind
            && let [name] = &path.segments
            && let Res::Local(id) = path.res
            && !expr.span.in_external_macro(cx.sess().source_map())
            && let ty = cx.typeck_results().expr_ty(left)
            && is_type_diagnostic_item(cx, ty, sym::PathBuf)
        {
            self.searcher = Some(PathbufPushSearcher {
                local_id: id,
                lhs_is_let: false,
                let_ty_span: None,
                init_val: *right,
                arg: None,
                name: name.ident.name,
                err_span: expr.span,
            });
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let Some(mut searcher) = self.searcher.take()
            && let StmtKind::Expr(expr) | StmtKind::Semi(expr) = stmt.kind
            && let ExprKind::MethodCall(name, self_arg, [arg_expr], _) = expr.kind
            && path_to_local_id(self_arg, searcher.local_id)
            && name.ident.name == sym::push
        {
            searcher.err_span = searcher.err_span.to(stmt.span);
            searcher.arg = Some(*arg_expr);
            searcher.display_err(cx);
        }
    }

    fn check_block_post(&mut self, _: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        self.searcher = None;
    }
}
