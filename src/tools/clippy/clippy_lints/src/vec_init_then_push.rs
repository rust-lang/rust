use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::{get_vec_init_kind, VecInitKind};
use clippy_utils::source::snippet;
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{get_parent_expr, path_to_local_id};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{
    BindingAnnotation, Block, Expr, ExprKind, HirId, Local, Mutability, PatKind, QPath, Stmt, StmtKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `push` immediately after creating a new `Vec`.
    ///
    /// If the `Vec` is created using `with_capacity` this will only lint if the capacity is a
    /// constant and the number of pushes is greater than or equal to the initial capacity.
    ///
    /// If the `Vec` is extended after the initial sequence of pushes and it was default initialized
    /// then this will only lint after there were at least four pushes. This number may change in
    /// the future.
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
    #[clippy::version = "1.51.0"]
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
    lhs_is_let: bool,
    let_ty_span: Option<Span>,
    name: Symbol,
    err_span: Span,
    found: u128,
    last_push_expr: HirId,
}
impl VecPushSearcher {
    fn display_err(&self, cx: &LateContext<'_>) {
        let required_pushes_before_extension = match self.init {
            _ if self.found == 0 => return,
            VecInitKind::WithConstCapacity(x) if x > self.found => return,
            VecInitKind::WithConstCapacity(x) => x,
            VecInitKind::WithExprCapacity(_) => return,
            _ => 3,
        };

        let mut needs_mut = false;
        let res = for_each_local_use_after_expr(cx, self.local_id, self.last_push_expr, |e| {
            let Some(parent) = get_parent_expr(cx, e) else {
                return ControlFlow::Continue(())
            };
            let adjusted_ty = cx.typeck_results().expr_ty_adjusted(e);
            let adjusted_mut = adjusted_ty.ref_mutability().unwrap_or(Mutability::Not);
            needs_mut |= adjusted_mut == Mutability::Mut;
            match parent.kind {
                ExprKind::AddrOf(_, Mutability::Mut, _) => {
                    needs_mut = true;
                    return ControlFlow::Break(true);
                },
                ExprKind::Unary(UnOp::Deref, _) | ExprKind::Index(..) if !needs_mut => {
                    let mut last_place = parent;
                    while let Some(parent) = get_parent_expr(cx, last_place) {
                        if matches!(parent.kind, ExprKind::Unary(UnOp::Deref, _) | ExprKind::Field(..))
                            || matches!(parent.kind, ExprKind::Index(e, _) if e.hir_id == last_place.hir_id)
                        {
                            last_place = parent;
                        } else {
                            break;
                        }
                    }
                    needs_mut |= cx.typeck_results().expr_ty_adjusted(last_place).ref_mutability()
                        == Some(Mutability::Mut)
                        || get_parent_expr(cx, last_place)
                            .map_or(false, |e| matches!(e.kind, ExprKind::AddrOf(_, Mutability::Mut, _)));
                },
                ExprKind::MethodCall(_, [recv, ..], _)
                    if recv.hir_id == e.hir_id
                        && adjusted_mut == Mutability::Mut
                        && !adjusted_ty.peel_refs().is_slice() =>
                {
                    // No need to set `needs_mut` to true. The receiver will be either explicitly borrowed, or it will
                    // be implicitly borrowed via an adjustment. Both of these cases are already handled by this point.
                    return ControlFlow::Break(true);
                },
                ExprKind::Assign(lhs, ..) if e.hir_id == lhs.hir_id => {
                    needs_mut = true;
                    return ControlFlow::Break(false);
                },
                _ => (),
            }
            ControlFlow::Continue(())
        });

        // Avoid allocating small `Vec`s when they'll be extended right after.
        if res == ControlFlow::Break(true) && self.found <= required_pushes_before_extension {
            return;
        }

        let mut s = if self.lhs_is_let {
            String::from("let ")
        } else {
            String::new()
        };
        if needs_mut {
            s.push_str("mut ");
        }
        s.push_str(self.name.as_str());
        if let Some(span) = self.let_ty_span {
            s.push_str(": ");
            s.push_str(&snippet(cx, span, "_"));
        }
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

impl<'tcx> LateLintPass<'tcx> for VecInitThenPush {
    fn check_block(&mut self, _: &LateContext<'tcx>, _: &'tcx Block<'tcx>) {
        self.searcher = None;
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'tcx>) {
        if let Some(init_expr) = local.init
            && let PatKind::Binding(BindingAnnotation::Mutable, id, name, None) = local.pat.kind
            && !in_external_macro(cx.sess(), local.span)
            && let Some(init) = get_vec_init_kind(cx, init_expr)
            && !matches!(init, VecInitKind::WithExprCapacity(_))
        {
            self.searcher = Some(VecPushSearcher {
                local_id: id,
                init,
                lhs_is_let: true,
                name: name.name,
                let_ty_span: local.ty.map(|ty| ty.span),
                err_span: local.span,
                found: 0,
                last_push_expr: init_expr.hir_id,
            });
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if self.searcher.is_none()
            && let ExprKind::Assign(left, right, _) = expr.kind
            && let ExprKind::Path(QPath::Resolved(None, path)) = left.kind
            && let [name] = &path.segments
            && let Res::Local(id) = path.res
            && !in_external_macro(cx.sess(), expr.span)
            && let Some(init) = get_vec_init_kind(cx, right)
            && !matches!(init, VecInitKind::WithExprCapacity(_))
        {
            self.searcher = Some(VecPushSearcher {
                local_id: id,
                init,
                lhs_is_let: false,
                let_ty_span: None,
                name: name.ident.name,
                err_span: expr.span,
                found: 0,
                last_push_expr: expr.hir_id,
            });
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let Some(searcher) = self.searcher.take() {
            if let StmtKind::Expr(expr) | StmtKind::Semi(expr) = stmt.kind
                && let ExprKind::MethodCall(name, [self_arg, _], _) = expr.kind
                && path_to_local_id(self_arg, searcher.local_id)
                && name.ident.as_str() == "push"
            {
                self.searcher = Some(VecPushSearcher {
                    found: searcher.found + 1,
                    err_span: searcher.err_span.to(stmt.span),
                    last_push_expr: expr.hir_id,
                    .. searcher
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
