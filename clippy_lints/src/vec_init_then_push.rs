use crate::utils::{is_type_diagnostic_item, match_def_path, paths, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BindingAnnotation, Block, Expr, ExprKind, Local, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{symbol::sym, Span, Symbol};
use std::convert::TryInto;

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `push` immediately after creating a new `Vec`.
    ///
    /// **Why is this bad?** The `vec![]` macro is both more performant and easier to read than
    /// multiple `push` calls.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
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

#[derive(Clone, Copy)]
enum VecInitKind {
    New,
    WithCapacity(u64),
}
struct VecPushSearcher {
    init: VecInitKind,
    name: Symbol,
    lhs_is_local: bool,
    lhs_span: Span,
    err_span: Span,
    found: u64,
}
impl VecPushSearcher {
    fn display_err(&self, cx: &LateContext<'_>) {
        match self.init {
            _ if self.found == 0 => return,
            VecInitKind::WithCapacity(x) if x > self.found => return,
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
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'tcx>) {
        self.searcher = None;
        if_chain! {
            if !in_external_macro(cx.sess(), local.span);
            if let Some(init) = local.init;
            if let PatKind::Binding(BindingAnnotation::Mutable, _, ident, None) = local.pat.kind;
            if let Some(init_kind) = get_vec_init_kind(cx, init);
            then {
                self.searcher = Some(VecPushSearcher {
                        init: init_kind,
                        name: ident.name,
                        lhs_is_local: true,
                        lhs_span: local.ty.map_or(local.pat.span, |t| local.pat.span.to(t.span)),
                        err_span: local.span,
                        found: 0,
                    });
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if self.searcher.is_none() {
            if_chain! {
                if !in_external_macro(cx.sess(), expr.span);
                if let ExprKind::Assign(left, right, _) = expr.kind;
                if let ExprKind::Path(QPath::Resolved(_, path)) = left.kind;
                if let Some(name) = path.segments.get(0);
                if let Some(init_kind) = get_vec_init_kind(cx, right);
                then {
                    self.searcher = Some(VecPushSearcher {
                        init: init_kind,
                        name: name.ident.name,
                        lhs_is_local: false,
                        lhs_span: left.span,
                        err_span: expr.span,
                        found: 0,
                    });
                }
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let Some(searcher) = self.searcher.take() {
            if_chain! {
                if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = stmt.kind;
                if let ExprKind::MethodCall(path, _, [self_arg, _], _) = expr.kind;
                if path.ident.name.as_str() == "push";
                if let ExprKind::Path(QPath::Resolved(_, self_path)) = self_arg.kind;
                if let [self_name] = self_path.segments;
                if self_name.ident.name == searcher.name;
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

fn get_vec_init_kind<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<VecInitKind> {
    if let ExprKind::Call(func, args) = expr.kind {
        match func.kind {
            ExprKind::Path(QPath::TypeRelative(ty, name))
                if is_type_diagnostic_item(cx, cx.typeck_results().node_type(ty.hir_id), sym::vec_type) =>
            {
                if name.ident.name == sym::new {
                    return Some(VecInitKind::New);
                } else if name.ident.name.as_str() == "with_capacity" {
                    return args.get(0).and_then(|arg| {
                        if_chain! {
                            if let ExprKind::Lit(lit) = &arg.kind;
                            if let LitKind::Int(num, _) = lit.node;
                            then {
                                Some(VecInitKind::WithCapacity(num.try_into().ok()?))
                            } else {
                                None
                            }
                        }
                    });
                }
            }
            ExprKind::Path(QPath::Resolved(_, path))
                if match_def_path(cx, path.res.opt_def_id()?, &paths::DEFAULT_TRAIT_METHOD)
                    && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::vec_type) =>
            {
                return Some(VecInitKind::New);
            }
            _ => (),
        }
    }
    None
}
